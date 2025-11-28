import os
import json
import re
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.genai as genai
from google.genai import types

# =========================
# Gemini API Key
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# =========================
# FastAPI
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
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

class Event(BaseModel):
    title: str
    date: str
    start_time: str
    end_time: str
    location: Optional[str] = ""
    notes: Optional[str] = ""
    raw_text: Optional[str] = None
    source: Optional[str] = "image"

class ParseScheduleResponse(BaseModel):
    events: List[Event]

# =========================
# ✅ 記憶體
# =========================
chat_memory: List[dict] = []
last_image_events: List[Event] = []   # ✅ 這是關鍵：記住最近一次圖片解析結果

# =========================
# 健康檢查
# =========================
@app.get("/")
async def root():
    return {"status": "ok"}

# =========================
# ✅ 聊天（支援「某一天行程」）
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    global last_image_events

    user_msg = req.message.strip()
    chat_memory.append({"role": "user", "content": user_msg})

    # ✅ 1️⃣ 先判斷是不是在問「某一天行程」
    match = re.search(r"(\d{1,2})\s*日", user_msg)

    if match and last_image_events:
        day = match.group(1).zfill(2)

        day_events = [
            e for e in last_image_events
            if e.date.endswith(f"-{day}")
        ]

        if not day_events:
            return ChatResponse(reply=f"📅 {int(day)} 日沒有任何行程")

        lines = [f"📅 {int(day)} 日行程："]
        for e in day_events:
            time = e.start_time or "--:--"
            title = e.title or e.notes or "未命名行程"
            lines.append(f"• {time} {title}")

        reply = "\n".join(lines)
        chat_memory.append({"role": "assistant", "content": reply})
        return ChatResponse(reply=reply)

    # ✅ 2️⃣ 一般自由聊天（像助理）
    system_prompt = {
        "role": "system",
        "content": "你是一個親切、簡潔、會用繁體中文回答的 AI 助理。"
    }

    messages = [system_prompt] + chat_memory[-10:]

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[m["content"] for m in messages]
        )
        reply = response.text.strip()
        chat_memory.append({"role": "assistant", "content": reply})
        return ChatResponse(reply=reply)

    except Exception as e:
        return ChatResponse(reply=f"❌ Gemini 錯誤：{str(e)}")

# =========================
# ✅ 圖片行事曆解析（會存入記憶體）
# =========================
@app.post("/parse-schedule-image", response_model=ParseScheduleResponse)
async def parse_schedule_image(image: UploadFile = File(...)):
    global last_image_events

    try:
        img_bytes = await image.read()

        prompt = """
你是一個行事曆 OCR 分析器，
請從圖片中擷取出所有「日期 + 時間 + 狀態（忙碌 / 暫定）」，
並輸出為 JSON 陣列，欄位如下：
title, date (YYYY-MM-DD), start_time (HH:MM), end_time

只輸出 JSON，不要解釋。
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=img_bytes,
                    mime_type=image.content_type or "image/jpeg"
                ),
                prompt,
            ],
        )

        raw_text = response.text.strip()

        data = json.loads(raw_text)

        events = [Event(**e) for e in data.get("events", [])]

        # ✅ 關鍵：存起來給之後查詢單日用
        last_image_events = events

        return ParseScheduleResponse(events=events)

    except Exception as e:
        return ParseScheduleResponse(events=[
            Event(
                title="解析失敗",
                date="",
                start_time="",
                end_time="",
                notes=str(e),
                source="image"
            )
        ])
