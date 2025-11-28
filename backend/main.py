import os
import json
import re
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.genai as genai
from google.genai import types
from datetime import datetime

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
# ✅ 聊天上下文
# =========================
chat_memory: List[dict] = []

# =========================
# Root
# =========================
@app.get("/")
async def root():
    return {"status": "ok"}

# =========================
# ✅ 一般聊天（助理模式）
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        chat_memory.append({"role": "user", "content": req.message})

        system_prompt = {
            "role": "system",
            "content": "你是溫暖、自然、會用繁體中文回答的 AI 助理，回答簡短、有條理。"
        }

        messages = [system_prompt] + chat_memory[-10:]

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[m["content"] for m in messages]
        )

        reply = response.text.strip()
        chat_memory.append({"role": "assistant", "content": reply})

        return ChatResponse(reply=reply)

    except Exception as e:
        return ChatResponse(reply=f"❌ 錯誤：{str(e)}")

# =========================
# ✅ 行事曆圖片 → 只輸出 Events JSON
# =========================
@app.post("/parse-schedule-image", response_model=ParseScheduleResponse)
async def parse_schedule_image(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()

        prompt = """
請從行事曆圖片中只萃取「行程資料」，
只回傳以下 JSON 陣列格式，不要任何說明：

[
  {
    "title": "",
    "date": "YYYY-MM-DD",
    "start_time": "HH:MM",
    "end_time": "",
    "location": "",
    "notes": "",
    "raw_text": null,
    "source": "image"
  }
]
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=img_bytes,
                    mime_type=image.content_type or "image/jpeg",
                ),
                prompt,
            ],
        )

        raw_text = response.text
        match = re.search(r"\[.*\]", raw_text, re.S)

        if not match:
            raise ValueError("無法擷取 JSON")

        events = json.loads(match.group(0))
        return ParseScheduleResponse(events=events)

    except Exception as e:
        return ParseScheduleResponse(events=[
            Event(
                title="解析失敗",
                date="",
                start_time="",
                end_time="",
                notes=str(e)
            )
        ])

# =========================
# ✅ 重點：指定某一天 → 極簡輸出格式
# =========================
@app.post("/get-day-schedule", response_model=ChatResponse)
async def get_day_schedule(
    target_date: str = Form(...),  # e.g. 2016-05-31
    events_json: str = Form(...)
):
    try:
        events = json.loads(events_json)

        filtered = [
            e for e in events
            if e.get("date") == target_date
        ]

        day = int(target_date.split("-")[2])

        if not filtered:
            return ChatResponse(reply=f"📅 {day} 日沒有行程")

        lines = [f"📅 {day} 日行程："]

        for e in filtered:
            time = e.get("start_time", "")
            title = e.get("title", "")
            lines.append(f"• {time} {title}")

        return ChatResponse(reply="\n".join(lines))

    except Exception as e:
        return ChatResponse(reply=f"❌ 行程整理失敗：{str(e)}")
