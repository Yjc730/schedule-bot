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
# ✅ 聊天上下文（簡單助理用）
# =========================
chat_memory: List[dict] = []

# =========================
# ✅ Root
# =========================
@app.get("/")
async def root():
    return {"status": "ok", "message": "Gemini API Running"}

# =========================
# ✅ 一般聊天（像助理一樣 Q&A）
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        system_prompt = """
你是一個溫暖、自然、會用繁體中文聊天的 AI 助手，
回答要「簡單、實用、像真人說話」，不要過度說明。
"""

        chat_memory.append({"role": "user", "content": req.message})
        messages = [{"role": "system", "content": system_prompt}] + chat_memory[-10:]

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[m["content"] for m in messages]
        )

        reply = response.text.strip()
        chat_memory.append({"role": "assistant", "content": reply})
        return ChatResponse(reply=reply)

    except Exception as e:
        return ChatResponse(reply=f"❌ 聊天錯誤：{str(e)}")

# =========================
# ✅ 行事曆圖片解析（「只回簡短重點版」）
# =========================
@app.post("/parse-schedule-image", response_model=ParseScheduleResponse)
async def parse_schedule_image(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()

        prompt = """
請從行事曆圖片中：
1️⃣ 只擷取「有事件的日期」
2️⃣ 只輸出「日期 + 開始時間 + 狀態」
3️⃣ 狀態只用：忙碌 / 暫定 / 空白
4️⃣ 嚴格輸出 JSON 格式：

[
  {
    "title": "行程",
    "date": "YYYY-MM-DD",
    "start_time": "HH:MM",
    "end_time": "",
    "location": "",
    "notes": "忙碌 或 暫定",
    "raw_text": null,
    "source": "image"
  }
]

❌ 不要輸出任何說明文字
❌ 不要描述畫面
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

        raw_text = response.text.strip()
        match = re.search(r"\[.*\]", raw_text, re.S)

        if not match:
            raise ValueError("AI 未回傳正確 JSON")

        events = json.loads(match.group(0))
        return ParseScheduleResponse(events=events)

    except Exception as e:
        return ParseScheduleResponse(events=[
            Event(
                title="圖片解析失敗",
                date="",
                start_time="",
                end_time="",
                location="",
                notes=str(e),
                raw_text=None,
                source="image"
            )
        ])
