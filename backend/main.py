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
# ✅ 記憶區（聊天 + 行事曆）
# =========================
chat_memory: List[dict] = []
schedule_memory: List[Event] = []

# =========================
# Root
# =========================
@app.get("/")
async def root():
    return {"status": "ok"}

# =========================
# ✅ 智能聊天（會自動判斷是否在問行事曆）
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        user_msg = req.message.strip()

        # ✅ 1️⃣ 如果使用者在問「某一天的行程」
        date_match = re.search(r"(\d{1,2})[ 日号]", user_msg)
        if date_match and schedule_memory:
            day = date_match.group(1).zfill(2)
            filtered = [
                e for e in schedule_memory if e.date.endswith(f"-{day}")
            ]

            if not filtered:
                return ChatResponse(reply=f"📭 {int(day)} 日目前沒有行程")

            result = f"📅 {int(day)} 日行程：\n"
            for e in filtered:
                result += f"• {e.start_time} {e.title}\n"

            return ChatResponse(reply=result.strip())

        # ✅ 2️⃣ 否則就是正常助理聊天
        chat_memory.append({"role": "user", "content": user_msg})
        system_prompt = {
            "role": "system",
            "content": "你是一個溫暖自然的繁體中文助理，回答要簡潔，不要長篇說明。"
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
        return ChatResponse(reply=f"❌ Gemini 錯誤：{str(e)}")

# =========================
# ✅ 圖片解析 → 真正轉成「乾淨的行事曆資料」
# =========================
@app.post("/parse-schedule-image", response_model=ParseScheduleResponse)
async def parse_schedule_image(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()

        prompt = """
請從圖片中辨識所有「行事曆行程」，
並嚴格只輸出以下格式的 JSON 陣列（不要說明）：

[
  {
    "title": "暫定 / 忙碌",
    "date": "YYYY-MM-DD",
    "start_time": "HH:MM",
    "end_time": "",
    "location": "",
    "notes": ""
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

        raw = response.text
        match = re.search(r"\[.*\]", raw, re.S)
        if not match:
            raise ValueError("AI 未回傳正確 JSON")

        events_data = json.loads(match.group(0))
        events = [Event(**e) for e in events_data]

        # ✅ 存入全域記憶，供之後「幾號有什麼行程」使用
        schedule_memory.clear()
        schedule_memory.extend(events)

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
