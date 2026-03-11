# ===== 標準函式庫 (Python built-in) =====
import os
import sys
import base64
import re
import json
import time
import random
from typing import Any, Dict, List, Optional, Union

# ===== 專案路徑設定 =====
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

# ===== 第三方套件 =====
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import google.genai as genai
from google.genai import types

# =========================
# Config
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "0").strip() == "1"
MAX_MEMORY_MESSAGES = int(os.getenv("MAX_MEMORY_MESSAGES", "30"))
RAG_FILE_SIZE_THRESHOLD = 1_000_000  

MODEL_FAST = os.getenv("MODEL_FAST", "gemini-2.5-flash").strip()   
MODEL_TEXT = os.getenv("MODEL_TEXT", "gemini-2.5-flash").strip()   

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# =========================
# FastAPI Setup
# =========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Response Models
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

class VoiceConfirmRequest(BaseModel):
    action: str
    slots: dict

class VoiceConfirmResponse(BaseModel):
    reply: str

# =========================
# Memory Management (多用戶隔離)
# =========================
chat_histories: Dict[str, List[Dict[str, Any]]] = {}

def get_user_memory(user_id: str) -> List[Dict[str, Any]]:
    if user_id not in chat_histories:
        chat_histories[user_id] = []
    return chat_histories[user_id]

def trim_memory(memory_list: list):
    if len(memory_list) > MAX_MEMORY_MESSAGES:
        memory_list[:] = memory_list[-MAX_MEMORY_MESSAGES:]

# =========================
# Prompts & Tools
# =========================
IMAGE_ANALYZE_PROMPT = """
你是一個圖片 / PDF 理解助理。請根據輸入內容，完成以下兩件事，並「只用 JSON 格式」回答：
{
  "type": "calendar | table | document | handwritten | ui | map | photo | other",
  "summary": "200 字以內的內容摘要"
}
"""

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

# 👇 新增這個：告警查詢跳轉工具
alarm_tool = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="open_alarm_system",
            description="當使用者想要查詢基地台告警代碼、Fault ID、或是詢問基地台故障、告警怎麼處理時，觸發此工具",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query_keyword": types.Schema(
                        type="STRING",
                        description="使用者想查詢的代碼或關鍵字（例如：7115-2、天線故障、查告警）"
                    )
                },
                required=["query_keyword"]
            )
        )
    ]
)

def web_search_tools() -> list:
    try:
        return [types.Tool(google_search=types.GoogleSearch())]
    except Exception:
        return []

def should_use_web_search(user_message: str) -> bool:
    triggers = ["最新", "新聞", "今天", "現在", "價格", "評價", "哪裡買", "維基", "規格", "上市", "發表", "誰是", "是什麼", "比較"]
    return any(t.lower() in user_message.lower() for t in triggers)

def make_part_from_bytes(data: bytes, mime: str) -> "types.Part":
    try:
        return types.Part.from_bytes(data=data, mime_type=mime)
    except TypeError:
        return types.Part.from_bytes(data, mime)

# =========================
# RAG Helpers (✅ 保留你原本的向量搜尋邏輯)
# =========================
rag_store = []  

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
        return [[0.0] * 768 for _ in texts]  
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
# API Endpoints
# =========================
@app.get("/")
async def root():
    return {"status": "ok", "message": "Gemini Streaming Assistant running"}

@app.post("/reset", response_model=ChatResponse)
async def reset():
    chat_histories.clear()
    return ChatResponse(reply="✅ 已清除所有後端記憶")

# =========================
# 🚀 效能大升級：極速串流直通車版 Chat API
# =========================
@app.post("/chat")
async def chat(
    message: str = Form(""),
    user_id: str = Form("guest"),
    image: List[UploadFile] = File(default_factory=list),
):
    current_memory = get_user_memory(user_id)
    if not client:
        return StreamingResponse(iter(["❌ 後端尚未設定 GEMINI_API_KEY"]), media_type="text/plain")

    message = (message or "").strip()
    current_turn_parts = []  # 用來存放這次上傳的「實體檔案與文字」

    # 👉 1. 極速處理：跳過耗時的預先摘要，直接把檔案準備好餵給模型
    if image and len(image) > 0:
        for img in image:
            file_bytes = await img.read()
            mime = (img.content_type or "").strip().lower() or "application/octet-stream"
            filename = img.filename.lower()

            # CSV 報表：瞬間文字裁切處理 (毫秒級)
            if mime == "text/csv" or filename.endswith(".csv"):
                try:
                    file_text = file_bytes.decode("utf-8", errors="ignore")
                    lines = file_text.split("\n")[:50] # 只取前50行預覽，避免超出 token
                    preview = "\n".join(lines)
                    current_turn_parts.append(f"【資料表 {img.filename} 內容預覽】\n{preview}")
                except Exception as e:
                    current_turn_parts.append(f"讀取 CSV 失敗：{e}")
            
            # 圖片或 PDF：轉成 Gemini 原生 Part 格式 (毫秒級，不再中途浪費時間問 AI)
            else:
                part = make_part_from_bytes(file_bytes, mime)
                current_turn_parts.append(part)

            # 歷史記憶庫「只存文字紀錄」，避免伺服器被圖檔撐爆記憶體
            current_memory.append({
                "role": "file",
                "content": f"（使用者上傳了檔案: {img.filename}）",
            })

    # 如果使用者只傳圖/檔，卻沒打字，自動補上指令
    if not message and current_turn_parts:
        message = "請幫我詳細分析並總結這些檔案的內容。"

    # 👉 2. 準備對話歷史
    current_memory.append({"role": "user", "content": message})
    trim_memory(current_memory)

    system = """
    你是網站內的 AI 助理，負責協助使用者完成實際任務，請使用繁體中文。
    重要規則：
    1. 當使用者上傳報表或圖片時，請化身專業分析師，給出排版清晰的洞察（多用 Markdown 表格和粗體）。
    2. 若使用者有寄信、請假意圖，請立刻呼叫工具準備信件。
    """
    convo = [system]
    
    # 塞入歷史文字對話 (抓取最近 10 筆，但不包含剛剛加的最新一筆)
    for m in current_memory[-10:-1]:
        if m["role"] == "user":
            convo.append(f"使用者：{m['content']}")
        elif m["role"] == "assistant":
            convo.append(f"助理：{m['content']}")
        elif m["role"] == "file":
            convo.append(m['content'])
    
    # 👉 3. 關鍵直通車：把「實體檔案」和「這回合的問題」一起丟給 AI
    convo.extend(current_turn_parts)
    convo.append(f"使用者：{message}")

    tools = [email_tool, alarm_tool]
    if ENABLE_WEB_SEARCH and should_use_web_search(message):
        web_tools = web_search_tools()
        if web_tools:
            tools.extend(web_tools)

    # 👉 4. 串流輸出 Generator
    async def stream_generator():
        try:
            # 這裡才是「唯一一次」發送 API，省時又省錢！
            response_stream = client.models.generate_content_stream(
                model=MODEL_TEXT,
                contents=convo,
                config=types.GenerateContentConfig(tools=tools)
            )

            full_reply = ""
            for chunk in response_stream:
                if chunk.function_calls:
                    fc = chunk.function_calls[0]
                    if fc.name == "extract_email_intent":
                        rec = fc.args.get("recipient_name", "")
                        body = fc.args.get("email_content", "")
                        sys_msg = f"📧 【系統動作：準備寄信】\n收件人：{rec}\n內容：{body}\n\n請問確認要寄出嗎？"
                        full_reply += sys_msg
                        yield sys_msg
                        break 

                    elif fc.name == "open_alarm_system":
                        keyword = fc.args.get("query_keyword", "未知代碼")
                        
                        # 發送帶有特殊暗號的字串給前端
                        sys_msg = f"🚨 【系統動作：開啟告警查詢】\n偵測到您想查詢告警代碼或故障資訊 ({keyword})。\n請點擊下方按鈕前往「Alarm-Fault 專屬系統」進行深度查詢："
                        yield sys_msg
                        break

                if chunk.text:
                    full_reply += chunk.text
                    yield chunk.text 

            if full_reply:
                current_memory.append({"role": "assistant", "content": full_reply})
                trim_memory(current_memory)

        except Exception as e:
            yield f"\n⚠️ 產生回應時發生錯誤：{str(e)}"

    return StreamingResponse(stream_generator(), media_type="text/plain")


# =========================
# Voice Confirm API (✅ 保留你原本的語音確認邏輯)
# =========================
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
            return VoiceConfirmResponse(reply="❌ 找不到收件人，已取消操作")

        recipient_email = CONTACTS.get(recipient_name)
        if not recipient_email:
            return VoiceConfirmResponse(reply=f"❌ 我不知道「{recipient_name}」是誰")

        try:
            send_email_via_outlook(recipient_email, body)
            return VoiceConfirmResponse(reply=f"✅ 已幫你寄信給「{recipient_name}」")
        except Exception as e:
            return VoiceConfirmResponse(reply=f"❌ 寄信失敗：{str(e)}")

    return VoiceConfirmResponse(reply="🤷 這個操作我還不會")

