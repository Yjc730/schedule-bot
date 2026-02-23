# ===== 標準函式庫 (Python built-in) =====
import os
import sys
import base64
import re
import json

from typing import Any, Dict, List, Optional, Union


# ===== 專案路徑設定 =====
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)


# ===== 第三方套件 =====
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
class VoiceCommandRequest(BaseModel):
    text: str

class VoiceCommandResponse(BaseModel):
    reply: str
    need_confirm: bool = True
    action: Optional[str] = None
    slots: Optional[dict] = None


# =========================
# Memory（圖片 + 對話 + PDF）
# =========================
"""
current_memory 結構：
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

chat_histories: Dict[str, List[Dict[str, Any]]] = {}

def get_user_memory(user_id: str) -> List[Dict[str, Any]]:
    """根據 user_id 取得對應的記憶，如果沒有就建立一個新的"""
    if user_id not in chat_histories:
        chat_histories[user_id] = []
    return chat_histories[user_id]

def trim_memory(memory_list: list):
    """
    確保傳入的對話清單不會超過最大長度限制。
    使用 memory_list[:] = ... 是為了原地修改(In-place)清單內容。
    """
    if len(memory_list) > MAX_MEMORY_MESSAGES:
        memory_list[:] = memory_list[-MAX_MEMORY_MESSAGES:]


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

IMAGE_ANALYZE_PROMPT = """
你是一個圖片 / PDF 理解助理。
請根據輸入內容，完成以下兩件事，並「只用 JSON 格式」回答：

{
  "type": "calendar | table | document | handwritten | ui | map | photo | other",
  "summary": "200 字以內的內容摘要"
}

規則：
- type 只能是列舉的其中一個英文值
- summary 請依內容客觀摘要
- 不要回答使用者問題
- 不要加入額外說明文字
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

# =========================
# AI Tools (Function Calling 工具定義)
# =========================
email_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="extract_email_intent",
            description="當使用者想要寄信、發電子郵件、聯絡某人時，提取收件人與內容",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "recipient_name": types.Schema(
                        type="STRING",
                        description="收件人的名稱或稱呼（例如：老闆、主管、王經理）"
                    ),
                    "email_content": types.Schema(
                        type="STRING",
                        description="信件的具體內容"
                    )
                },
                required=["recipient_name", "email_content"]
            )
        )
    ]
)

def parse_voice_intent(text: str) -> dict:
    """
    進階版：使用 LLM Function Calling 自動分析語音意圖
    """
    if not client:
        return {"intent": "chat", "slots": {}}

    try:
        # 讓 Gemini 自己判斷這句話是不是要寄信
        res = client.models.generate_content(
            model=MODEL_FAST,
            contents=f"請分析這句話是否有寄信意圖，如果有，請呼叫工具提取資訊：\n使用者說：「{text}」",
            config=types.GenerateContentConfig(
                tools=[email_tool],
                temperature=0.1  # 溫度調低，讓提取更精準
            )
        )
        
        # 如果 LLM 決定呼叫工具，代表它判斷這是一個寄信動作！
        if res.function_calls:
            fc = res.function_calls[0]
            if fc.name == "extract_email_intent":
                return {
                    "intent": "send_email",
                    "slots": {
                        "recipient": fc.args.get("recipient_name", ""),
                        "body": fc.args.get("email_content", "")
                    }
                }
    except Exception as e:
        print(f"意圖分析失敗：{e}")
        
    # 如果沒有呼叫工具，或者發生錯誤，就當作一般聊天
    return {
        "intent": "chat",
        "slots": {}
    }

import time
import random

def safe_generate_content(
    *,
    model: str,
    contents: list,
    tools: Optional[list] = None,
    max_retry: int = 3
) -> str:
    """
    統一處理 Gemini 呼叫錯誤（429 / 503 / overloaded）
    含自動 retry + backoff
    """
    for attempt in range(max_retry):
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

            if res.function_calls:
                fc = res.function_calls[0]
                if fc.name == "extract_email_intent":
                    rec = fc.args.get("recipient_name", "")
                    body = fc.args.get("email_content", "")
                    return f"📧 【系統動作：準備寄信】\n收件人：{rec}\n內容：{body}\n\n請問確認要寄出嗎？（若確認，請由前端觸發寄信邏輯）"

        return (res.text or "").strip()

        except Exception as e:
            msg = str(e)

            # ===== 過載 / 暫時不可用（最常見）=====
            if (
                "503" in msg
                or "UNAVAILABLE" in msg
                or "overloaded" in msg.lower()
            ):
                if attempt < max_retry - 1:
                    # 指數退避 + jitter
                    sleep_time = 1.2 * (attempt + 1) + random.uniform(0, 0.6)
                    time.sleep(sleep_time)
                    continue
                return "⚠️ 模型目前繁忙，請稍後再試（系統已自動重試）"

            # ===== 額度用盡 =====
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                return "⚠️ 目前 Gemini 額度或流量已達上限，請稍後再試或更換模型。"

            # ===== 其他錯誤 =====
            return f"⚠️ Gemini 呼叫失敗：{msg}"

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
    if not client:
        return [[0.0] * 768 for _ in texts]  # dummy embedding，避免 crash
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
    current_memory.clear()
    return ChatResponse(reply="✅ 已清除後端記憶（current_memory）")


# =========================
# Chat API (multipart: message + image/pdf)
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(
    message: str = Form(""),
    user_id: str = Form("guest"),  # 👈 新增這一行，預設值給 guest
    image: List[UploadFile] = File(default_factory=list),
):
    # 在函數的第一行，先取得該使用者的專屬記憶體
    current_memory = get_user_memory(user_id)
    if not client:
        return ChatResponse(reply="❌ 後端尚未設定 GEMINI_API_KEY")

    message = (message or "").strip()

    # =========================
    # Case 1：有上傳檔案（多檔）
    # =========================
    if image and len(image) > 0:
        summaries = []

        for img in image:
            file_bytes = await img.read()
            mime = (img.content_type or "").strip().lower() or "application/octet-stream"

            use_rag = (
                mime == "application/pdf"
                and len(file_bytes) > RAG_FILE_SIZE_THRESHOLD
            )

            file_b64 = base64.b64encode(file_bytes).decode("utf-8")
            part = make_part_from_bytes(file_bytes, mime)

            result = safe_generate_content(
                model=MODEL_FAST,
                contents=[IMAGE_ANALYZE_PROMPT, part],
            )

            try:
                parsed = json.loads(result)
                file_type = parsed.get("type", "other").lower()
                file_summary = parsed.get("summary", "")
            except Exception:
                file_type = "other"
                file_summary = result

            current_memory.append({
                "role": "file",
                "content": message or "[使用者上傳檔案]",
                "filename": img.filename or "uploaded",
                "b64": file_b64,
                "mime": mime,
                "summary": file_summary,
                "type": file_type,
            })

            summaries.append({
                "filename": img.filename,
                "summary": file_summary,
                "type": file_type,
                "use_rag": use_rag
            })

        trim_memory(current_memory)

        # --- 有問題 → 跨檔回答 ---
        if message:
            all_summaries = "\n\n".join(
                f"【{s['filename']}】\n{s['summary']}"
                for s in summaries
            )

            reply = safe_generate_content(
                model=MODEL_TEXT,
                contents=[
                    "你正在同時閱讀多個檔案，請整合所有資訊後回答。",
                    f"【檔案摘要集合】\n{all_summaries}",
                    f"使用者問題：{message}"
                ]
            )

            current_memory.append({"role": "assistant", "content": reply})
            trim_memory(current_memory)
            return ChatResponse(reply=reply)

        # --- 沒問題 → 回摘要 ---
        return ChatResponse(
            reply=(
                f"✅ 已讀取 {len(summaries)} 個檔案。\n\n"
                + "\n\n".join(
                    f"【{s['filename']}】\n{s['summary']}"
                    for s in summaries
                )
            )
        )

    # =========================
    # Case 2：純文字聊天（沒有檔案）
    # =========================
    if not message:
        return ChatResponse(reply="請輸入問題或上傳圖片 / PDF 喔！")

    current_memory.append({"role": "user", "content": message})
    trim_memory(current_memory)

    system = """
你是網站內的 AI 助理，負責協助使用者完成實際任務，請使用繁體中文。

重要規則（請嚴格遵守）：
1. 你不需要也不應該說「我無法寄信」、「我不能開 Outlook」。
2. 當你已經產生完整的請假信或信件內容時：
   - 請直接提供完整信件內容。
   - 可以詢問使用者是否要「用 Outlook 寄出」。
3. 因為前端介面已提供「用 Outlook 寄出」按鈕，實際寄送會由使用者在 Outlook 中完成。
4. 請把你的角色視為「幫助完成寄信流程的一部分」，而不是單純聊天機器人。
"""


    convo = [system]
    for m in current_memory[-10:]:
        if m["role"] == "user":
            convo.append(f"使用者：{m['content']}")
        elif m["role"] == "assistant":
            convo.append(f"助理：{m['content']}")
        elif m["role"] == "file":
            convo.append(f"（先前檔案摘要）：{m.get('summary','')}")

    convo.append(f"使用者：{message}")

    # 預設把寄信工具交給 AI
    tools = [email_tool]
    if ENABLE_WEB_SEARCH and should_use_web_search(message):
        web_tools = web_search_tools()
        if web_tools:
            tools.extend(web_tools) # 如果有開啟搜尋，就把搜尋工具也加進去

    reply = safe_generate_content(
        model=MODEL_TEXT,
        contents=convo,
        tools=tools,
    )

    current_memory.append({"role": "assistant", "content": reply})
    trim_memory(current_memory)
    return ChatResponse(reply=reply)

# =========================
# Shared Core (Text-only)
# 給 CLI / Voice Agent / 未來 WebSocket 用
# 不影響現有 /chat API
# =========================
def handle_text_query(message: str) -> str:
    current_memory = get_user_memory("guest")
    """
    純文字查詢入口（不含圖片 / PDF）
    - 共用既有 current_memory
    - 共用 Gemini 設定、工具、web search
    - 不動 FastAPI /chat 行為
    """

    if not client:
        return "❌ 後端尚未設定 GEMINI_API_KEY"

    message = (message or "").strip()
    if not message:
        return "請輸入問題喔！"

    # ---- 記憶：user ----
    current_memory.append({"role": "user", "content": message})
    trim_memory(current_memory)

    system = """
    你是網站內的 AI 助理，負責協助使用者完成實際任務，請使用繁體中文。
    
    重要規則（請嚴格遵守）：
    1. 你不需要也不應該說「我無法寄信」、「我不能開 Outlook」。
    2. 當你已經產生完整的請假信或信件內容時：
       - 請直接提供完整信件內容。
       - 可以詢問使用者是否要「用 Outlook 寄出」。
    3. 因為前端介面已提供「用 Outlook 寄出」按鈕，實際寄送會由使用者在 Outlook 中完成。
    4. 請把你的角色視為「幫助完成寄信流程的一部分」，而不是單純聊天機器人。
    口語化、但回答有重點的 AI 助理，請使用繁體中文。
    """

    convo = [system]

    # ---- 最近對話記憶 ----
    for m in current_memory[-10:]:
        if m["role"] == "user":
            convo.append(f"使用者：{m['content']}")
        elif m["role"] == "assistant":
            convo.append(f"助理：{m['content']}")
        elif m["role"] == "file":
            convo.append(f"（先前檔案摘要）：{m.get('summary','')}")

    convo.append(f"使用者：{message}")

    # 預設把寄信工具交給 AI
    tools = [email_tool]
    if ENABLE_WEB_SEARCH and should_use_web_search(message):
        web_tools = web_search_tools()
        if web_tools:
            tools.extend(web_tools) # 如果有開啟搜尋，就把搜尋工具也加進去

    reply = safe_generate_content(
        model=MODEL_TEXT,
        contents=convo,
        tools=tools,
    )

    # ---- 記憶：assistant ----
    current_memory.append({"role": "assistant", "content": reply})
    trim_memory(current_memory)

    return reply


@app.post("/voice-command", response_model=VoiceCommandResponse)
async def voice_command(req: VoiceCommandRequest):
    text = (req.text or "").strip()
    if not text:
        return VoiceCommandResponse(
            reply="我沒有聽清楚，可以再說一次嗎？",
            need_confirm=False
        )

    intent_data = parse_voice_intent(text)
    intent = intent_data["intent"]
    slots = intent_data.get("slots", {})

    # ===== 寄信流程（確認階段）=====
    if intent == "send_email":
        recipient = slots.get("recipient") or "對方"
        body = slots.get("body") or ""

        reply = (
            f"你要寄信給「{recipient}」，"
            f"內容是「{body}」，對嗎？"
        )

        return VoiceCommandResponse(
            reply=reply,
            need_confirm=True,
            action="send_email",
            slots=slots
        )

    # ===== 不是動作 → 當一般聊天 =====
    reply = handle_text_query(text)
    return VoiceCommandResponse(
        reply=reply,
        need_confirm=False
    )

class VoiceConfirmRequest(BaseModel):
    action: str
    slots: dict

class VoiceConfirmResponse(BaseModel):
    reply: str

from actions.send_email import send_email_via_outlook

CONTACTS = {
    "主管": "boss@example.com",
    "老闆": "boss@example.com",
}

@app.post("/voice-confirm", response_model=VoiceConfirmResponse)
async def voice_confirm(req: VoiceConfirmRequest):
    action = req.action
    slots = req.slots or {}

    # ===== 寄信 =====
    if action == "send_email":
        recipient_name = slots.get("recipient")
        body = slots.get("body", "")

        if not recipient_name:
            return VoiceConfirmResponse(
                reply="❌ 找不到收件人，已取消操作"
            )

        recipient_email = CONTACTS.get(recipient_name)
        if not recipient_email:
            return VoiceConfirmResponse(
                reply=f"❌ 我不知道「{recipient_name}」是誰"
            )

        try:
            send_email_via_outlook(recipient_email, body)
            return VoiceConfirmResponse(
                reply=f"✅ 已幫你寄信給「{recipient_name}」"
            )
        except Exception as e:
            return VoiceConfirmResponse(
                reply=f"❌ 寄信失敗：{str(e)}"
            )

    return VoiceConfirmResponse(
        reply="🤷 這個操作我還不會"
    )
# =========================
# (NO router needed here)
# voice APIs are defined in this file
# =========================












