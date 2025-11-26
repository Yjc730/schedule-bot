import os
import json
import re
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types

# =========================
# ✅ 正確的環境變數方式
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

# ✅ 初始化 Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

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
# Root
# =========================
@app.get("/")
async def root():
    return {"status": "ok", "message": "Gemini AI API Running"}

# =========================
# ✅ 聊天（Gemini 2.5 Flash）
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=req.message
        )
        return ChatResponse(reply=response.text)
    except Exception as e:
        return ChatResponse(reply=f"Gemini 錯誤：{str(e)}")

# =========================
# ✅ 圖片解析（Gemini Vision）
# =========================
@app.post("/parse-schedule-image", response_model=ParseScheduleResponse)
async def parse_schedule_image(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()

        prompt = """
請從圖片中辨識所有行程，並輸出為 JSON 陣列：
[{
  "title": "",
  "date": "YYYY-MM-DD",
  "start_time": "HH:MM",
  "end_time": "HH:MM",
  "location": "",
  "notes": "",
  "raw_text": null,
  "source": "image"
}]
⚠️ 僅回傳 JSON，不要加說明文字。
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=img_bytes,
                    mime_type=image.content_type or "image/jpeg",
                ),
                prompt
            ]
        )

        raw_text = response.text
        match = re.search(r"\[.*\]", raw_text, re.S)

        if not match:
            raise ValueError(f"非 JSON 回傳：{raw_text}")

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
