import os
import re
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.genai as genai
from google.genai import types

# =========================
# Gemini
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
    start_time: Optional[str] = ""
    end_time: Optional[str] = ""
    status: Optional[str] = ""
    calendar_type: Optional[str] = "image"
    location: Optional[str] = ""
    notes: Optional[str] = ""
    raw_text: Optional[str] = None
    source: Optional[str] = "image"

class ParseScheduleResponse(BaseModel):
    events: List[Event]

# =========================
# ✅ 記憶區
# =========================
chat_memory: List[dict] = []
event_memory: List[Event] = []

# =========================
# ✅ Chat（同時支援問答 + 查行程）
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    user_text = req.message.strip()

    # ✅ 嘗試抓「幾號」
    day_match = re.search(r"(\d{1,2})\s*日", user_text)

    if day_match:
        day = day_match.group(1).zfill(2)

        day_events = [
            e for e in event_memory
            if e.date.endswith(f"-{day}")
        ]

        if not day_events:
            return ChatResponse(reply=f"📅 {int(day)} 日沒有任何行程")

        lines = [f"📅 {int(day)} 日行程："]
        for e in day_events:
            status = e.status or e.title or "行程"
            time = e.start_time or "未知時間"
            lines.append(f"• {time} {status}")

        return ChatResponse(reply="\n".join(lines))

    # ✅ 一般聊天（像助理）
    chat_memory.append({"role": "user", "content": user_text})

    system_prompt = {
        "role": "system",
        "content": "你是自然親切的 AI 助理，使用繁體中文回答。"
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
        return ChatResponse(reply=f"❌ 發生錯誤：{str(e)}")

# =========================
# ✅ 圖片 → 結構化行程
# =========================
@app.post("/parse-schedule-image", response_model=ParseScheduleResponse)
async def parse_schedule_image(image: UploadFile = File(...)):
    img_bytes = await image.read()

    prompt = """
請將這張圖片中的「所有行事曆行程」轉為 JSON 陣列，
格式如下：

[
  {
    "title": "會議",
    "date": "2025-01-31",
    "start_time": "09:30",
    "end_time": "10:00",
    "status": "暫定",
    "location": "",
    "notes": ""
  }
]

只輸出 JSON，不要輸出說明文字。
"""

    try:
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

        json_text = response.text.strip()
        events_data = json.loads(json_text)

        parsed_events = []
        for e in events_data:
            event = Event(**e, source="image")
            parsed_events.append(event)
            event_memory.append(event)

        return ParseScheduleResponse(events=parsed_events)

    except Exception as e:
        return ParseScheduleResponse(events=[
            Event(
                title="解析失敗",
                date="",
                notes=str(e),
                source="image"
            )
        ])
