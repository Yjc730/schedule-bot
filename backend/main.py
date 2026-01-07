import os
import base64
import re
from typing import List, Dict, Optional, Union, Any

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.genai as genai
from google.genai import types


# =========================
# Config
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "0").strip() == "1"  # 模型自動判斷需要才搜尋
MAX_MEMORY_MESSAGES = int(os.getenv("MAX_MEMORY_MESSAGES", "30"))
# =========================
# RAG Config
# =========================
RAG_FILE_SIZE_THRESHOLD = 1_000_000  # 1MB 以上 PDF 啟用 RAG

# 模型可依你的額度/需求調整
MODEL_FAST = os.getenv("MODEL_FAST", "gemini-2.5-flash").strip()   # 圖片分類/摘要/一般聊天
MODEL_TEXT = os.getenv("MODEL_TEXT", "gemini-2.5-flash").strip()   # 純文字聊天（可同 MODEL_FAST）

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# =========================
# FastAPI
# =========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 若你要更安全可改成你的前端 domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Response Model
# =========================
class ChatResponse(BaseModel):
    reply: str


# =========================
# Memory（圖片 + 對話 + PDF）
# =========================
"""
chat_memory 結構：
[
  { "role": "user", "content": "問題" },
  { "role": "assistant", "content": "回答" },
  {
    "role": "file",
    "content": "使用者上傳檔案時的文字描述",
    "filename": "...",
    "b64": "...",
    "mime": "image/jpeg" | "application/pdf",
    "summary": "檔案摘要",
    "type": "calendar/table/document/ui/map/photo/other"
  }
]
"""
chat_memory: List[Dict[str, Any]] = []


def trim_memory():
    """避免記憶無限膨脹（保留最後 MAX_MEMORY_MESSAGES 筆）"""
    global chat_memory
    if len(chat_memory) > MAX_MEMORY_MESSAGES:
        chat_memory = chat_memory[-MAX_MEMORY_MESSAGES:]


# =========================
# Prompts
# =========================
RAG_ANSWER_PROMPT = """
你是一個文件問答助理。
請只根據以下「文件片段」回答問題。
- 若文件中找不到答案，請明確回答「文件中未提及」
- 不要自行推測
- 回答請簡潔、有根據
"""

IMAGE_CLASSIFY_PROMPT = """
你是一個圖片/頁面類型分類助手。
請判斷這份內容（可能是圖片或 PDF 頁面）屬於以下哪一種：
- 行事曆（calendar）
- 表格（table）
- 文件（document）
- 手寫筆記（handwritten）
- UI 介面（ui）
- 地圖（map）
- 生活照片（photo）
- 其他（other）
只輸出一個類型字串（例如：calendar）
"""

FILE_SUMMARY_PROMPT = """
請將圖片或 PDF 的主要內容摘要成 200 字以內。
請包含：
- 主要物件或文字
- 若是文件，摘要段落重點
- 若是表格，摘要欄位與關鍵資料
- 若是行事曆，摘要有哪些事件（不要逐格描述）
- 若是照片，描述主要場景
禁止：
- 不要回答使用者問題
- 不要推測意圖
- 不要延伸評論
"""

def build_file_answer_prompt(file_type: str) -> str:
    base = """
你是一個「圖片/文件理解助理」。請根據使用者的問題，
只回答與問題直接相關的資訊。請使用繁體中文。
"""
    if file_type == "calendar":
        base += """
⚠️ 規則：
1. 僅回答問題涉及的日期、節日、事件或時段。
2. 不要列整個月/整份行事曆。
3. 若找不到精準答案，請回答「行事曆中沒有找到這個資訊」。
"""
    elif file_type == "table":
        base += """
⚠️ 規則：
1. 根據摘要回答欄位/數據。
2. 不要輸出整張表格。
"""
    elif file_type == "document":
        base += """
⚠️ 規則：
1. 根據摘要內容回答。
2. 回答務必簡潔、抓重點。
"""
    else:
        base += """
⚠️ 規則：
1. 僅依摘要內容回答。
2. 若摘要沒有，請回答「我在檔案摘要中沒有看到相關資訊」。
"""
    return base


# =========================
# Gemini helpers
# =========================
def make_part_from_bytes(data: bytes, mime: str) -> "types.Part":
    """
    修正你遇到的 Part.from_bytes 參數差異問題：
    有些版本是 from_bytes(data=..., mime_type=...)
    有些版本允許 from_bytes(data, mime)
    """
    try:
        return types.Part.from_bytes(data=data, mime_type=mime)
    except TypeError:
        # fallback：舊版/不同簽名
        return types.Part.from_bytes(data, mime)


def safe_generate_content(*, model: str, contents: list, tools: Optional[list] = None) -> str:
    """
    統一處理 Gemini 呼叫錯誤（含 429 quota）
    """
    try:
        if tools:
            res = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(tools=tools),
            )
        else:
            res = client.models.generate_content(
                model=model,
                contents=contents,
            )
        return (res.text or "").strip()
    except Exception as e:
        msg = str(e)
        # 你截圖的 429 RESOURCE_EXHAUSTED
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
            return "⚠️ 目前 Gemini 額度/流量已達上限（429 RESOURCE_EXHAUSTED）。請稍後再試，或升級方案/更換可用模型。"
        return f"⚠️ 伺服器呼叫 Gemini 失敗：{msg}"
# =========================
# RAG Helpers
# =========================
import numpy as np

rag_store = []  # 暫存向量（先用記憶體，不破壞現有架構）


def chunk_text(text: str, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def embed_texts(texts):
    res = client.models.embed_content(
        model="text-embedding-004",
        content=texts
    )
    return [e.values for e in res.embeddings]


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def add_to_rag_store(chunks, embeddings):
    for c, e in zip(chunks, embeddings):
        rag_store.append({
            "text": c,
            "embedding": e
        })


def retrieve_relevant_chunks(query, top_k=3):
    query_emb = embed_texts([query])[0]
    scored = []
    for item in rag_store:
        score = cosine_similarity(query_emb, item["embedding"])
        scored.append((score, item["text"]))
    scored.sort(reverse=True)
    return [t for _, t in scored[:top_k]]


# =========================
# Web search (Gemini tool)
# =========================
def should_use_web_search(user_message: str) -> bool:
    """
    第一層：快速 heuristic，避免每次都觸發工具（省額度、加快速度）。
    真正「模型自動判斷」會在第二層 prompt 再決策。
    """
    triggers = [
        "最新", "新聞", "今天", "現在", "價格", "評價", "哪裡買", "維基",
        "規格", "上市", "發表", "誰是", "是什麼", "為什麼大家說", "比較",
        "VW", "golf", "BMW", "i4", "VAG"
    ]
    return any(t.lower() in user_message.lower() for t in triggers)


def web_search_tools() -> list:
    """
    Gemini 官方 Web Search Tool（google_search）。
    注意：需要你的專案/帳號支援該 tool，否則會報錯。
    """
    try:
        return [types.Tool(google_search=types.GoogleSearch())]
    except Exception:
        # 如果你的 google-genai 版本沒有 GoogleSearch 類別，就先不啟用
        return []


# =========================
# Health Check
# =========================
@app.get("/")
async def root():
    return {"status": "ok", "message": "Gemini multimodal assistant running"}


# =========================
# (Optional) Reset memory
# =========================
@app.post("/reset", response_model=ChatResponse)
async def reset():
    chat_memory.clear()
    return ChatResponse(reply="✅ 已清除後端記憶（chat_memory）")


# =========================
# Chat API (multipart: message + image/pdf)
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(
    message: str = Form(""),
    image: Optional[UploadFile] = File(None),  # 前端欄位仍叫 image，但可上傳 image/* 或 application/pdf
):
    if not client:
        return ChatResponse(reply="❌ 後端尚未設定 GEMINI_API_KEY")

    message = (message or "").strip()

    # =========================
    # Case 1：有上傳檔案（圖片或 PDF）
    # =========================
    if image is not None:
        file_bytes = await image.read()
        mime = (image.content_type or "").strip().lower() or "application/octet-stream"
        use_rag = False
        if mime == "application/pdf" and len(file_bytes) > RAG_FILE_SIZE_THRESHOLD:
            use_rag = True


        # 存 base64（你未來想做「再次引用原檔」也方便）
        file_b64 = base64.b64encode(file_bytes).decode("utf-8")

        # (A) 類型判斷
        part = make_part_from_bytes(file_bytes, mime)
        file_type = safe_generate_content(
            model=MODEL_FAST,
            contents=[IMAGE_CLASSIFY_PROMPT, part],
        ).strip().lower()

        # 避免模型亂回：只收斂到這幾種
        allowed = {"calendar", "table", "document", "handwritten", "ui", "map", "photo", "other"}
        if file_type not in allowed:
            file_type = "other"

        # (B) 摘要
        file_summary = safe_generate_content(
            model=MODEL_FAST,
            contents=[FILE_SUMMARY_PROMPT, part],
        )

        # (C) 記憶
        chat_memory.append({
            "role": "file",
            "content": message or "[使用者上傳檔案]",
            "filename": image.filename or "uploaded",
            "b64": file_b64,
            "mime": mime,
            "summary": file_summary,
            "type": file_type,
        })
        trim_memory()

        # (D) 若有提問 → 依規則回答（只用「最近一份檔案摘要」避免你抱怨它列出整個月）
        # (D) 若有提問 → 回答
if message:

    # ===== RAG 路徑（只在大 PDF）=====
    if use_rag:
        chunks = chunk_text(file_summary)  # v1 先用摘要，穩
        embeddings = embed_texts(chunks)
        add_to_rag_store(chunks, embeddings)

        relevant = retrieve_relevant_chunks(message)

        reply = safe_generate_content(
            model=MODEL_TEXT,
            contents=[
                RAG_ANSWER_PROMPT,
                "【文件片段】\n" + "\n---\n".join(relevant),
                f"使用者問題：{message}"
            ]
        )

    # ===== 原本 summary QA（完全不動）=====
    else:
        prompt = build_file_answer_prompt(file_type)
        convo = [prompt]

        for m in reversed(chat_memory):
            if m.get("role") == "file":
                convo.append(f"【檔案摘要】：{m.get('summary','')}")
                break

        convo.append(f"使用者問題：{message}")

        reply = safe_generate_content(
            model=MODEL_FAST,
            contents=convo,
        )

    chat_memory.append({"role": "assistant", "content": reply})
    trim_memory()
    return ChatResponse(reply=reply)


        # 沒問問題：回摘要
        return ChatResponse(reply=f"✅ 已收到檔案（{image.filename}）。我整理的摘要如下：\n\n{file_summary}")

    # =========================
    # Case 2：純文字聊天（保留上下文 + 可選自動 web search）
    # =========================
    if not message:
        return ChatResponse(reply="請輸入問題或上傳圖片 / PDF 喔！")

    chat_memory.append({"role": "user", "content": message})
    trim_memory()

    system = """
你是一個自然、口語化、但回答有重點的 AI 助理，請使用繁體中文。
你會看到一段「對話記憶」與（可能的）「檔案摘要」。
- 如果使用者問的是先前上傳的圖片/PDF內容，優先用檔案摘要回答。
- 如果問題明顯需要最新/外部資訊，且你有 web search 工具可用，才使用搜尋。
- 若不需要搜尋，請不要搜尋。
"""

    # 組合對話上下文（含最近的檔案摘要）
    convo = [system]
    for m in chat_memory[-10:]:
        if m["role"] == "user":
            convo.append(f"使用者：{m['content']}")
        elif m["role"] == "assistant":
            convo.append(f"助理：{m['content']}")
        elif m["role"] == "file":
            convo.append(f"（先前檔案摘要）：{m.get('summary','')}")
    convo.append(f"使用者：{message}")

    # 是否啟用 web search 工具（不破壞現有功能：關掉就是原本行為）
    tools = None
    if ENABLE_WEB_SEARCH and should_use_web_search(message):
        tools = web_search_tools()
        # 若 tools 建立失敗，就退回不使用工具
        if not tools:
            tools = None

    reply = safe_generate_content(
        model=MODEL_TEXT,
        contents=convo,
        tools=tools,
    )

    chat_memory.append({"role": "assistant", "content": reply})
    trim_memory()
    return ChatResponse(reply=reply)

