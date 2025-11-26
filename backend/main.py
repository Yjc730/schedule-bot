import os
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# =========================
# OpenRouter 設定
# =========================

API_KEY = os.getenv("API_KEY")  # 一定要在 Railway Variables 設定
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "qwen/qwen-2.5-7b-instruct"  # 免費可用模型之一

# =========================
# FastAPI App
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
# Models
# =========================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    context: Optional[List[ChatMessage]] = None

class ChatResponse(BaseModel):
    reply: str

class ParseTextRequest(BaseModel):
    text: str
    reference_date: Optional[str] = None  # YYYY-MM-DD

class Event(BaseModel):
    title: str
    date: str        # YYYY-MM-DD
    start_time: str  # HH:MM
    end_time: str    # HH:MM
    location: Optional[str] = ""
    notes: Optional[str] = ""
    raw_text: Optional[str] = None
    source: Optional[str] = "text"

class ParseScheduleResponse(BaseModel):
    events: List[Event]

# =========================
# Root
# =========================

@app.get("/")
async def root():
    return {"status": "ok", "message": "Schedule Bot API"}

# =========================
# 真．LLM 聊天（OpenRouter）
# =========================

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not API_KEY:
        return ChatResponse(reply="❌ 伺服器尚未設定 API_KEY")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://your-app.example",  # 可隨便填
        "X-Title": "schedule-bot"
    }

    messages = [
        {"role": "system", "content": "你是一個友善、專業的繁體中文 AI 助手。"}
    ]

    if req.context:
        for msg in req.context:
            messages.append({
                "role": msg.role,
                "content": msg.content
            })

    messages.append({
        "role": "user",
        "content": req.message
    })

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.7,
    }

    try:
        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        reply_text = data["choices"][0]["message"]["content"]
        return ChatResponse(reply=reply_text)

    except Exception as e:
        return ChatResponse(reply=f"❌ 呼叫 LLM 失敗：{str(e)}")

# =========================
# 解析文字行程（暫時還是示範版）
# =========================

@app.post("/parse-schedule-text", response_model=ParseScheduleResponse)
async def parse_schedule_text(req: ParseTextRequest):
    dummy_event = Event(
        title="打掃",
        date="2025-11-25",
        start_time="13:00",
        end_time="14:00",
        location="家裡",
        notes="這是示範用假資料，之後可改成 LLM 解析",
        raw_text=req.text,
        source="text",
    )
    return ParseScheduleResponse(events=[dummy_event])

# =========================
# 解析圖片行程（暫時示範版）
# =========================

@app.post("/parse-schedule-image", response_model=ParseScheduleResponse)
async def parse_schedule_image(
    image: UploadFile = File(...),
    reference_date: Optional[str] = Form(None),
):
    filename = image.filename
    dummy_event = Event(
        title=f"從圖片（{filename}）讀到的假行程",
        date="2025-11-26",
        start_time="10:00",
        end_time="11:00",
        location="不明地點",
        notes="這是示範用假資料，之後可改成 Vision LLM",
        raw_text=None,
        source="image",
    )
    return ParseScheduleResponse(events=[dummy_event])
