import os
import json
import re
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.generativeai as genai

# =========================
# Gemini API Key（從 Railway 環境變數讀）
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

model_chat = genai.GenerativeModel("gemini-2.5-flash")
model_vision = genai.GenerativeModel("gemini-2.5-flash")

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
# Root
# =========================
@app.get("/")
async def root():
    return {"status": "ok", "message": "Gemini API Running"}

# =========================
# ✅ 聊天（Gemini 2.5 Flash）
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    resp = model_chat.generate_content(req.message)
    return ChatResponse(reply=resp.text)

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
⚠️ 只回傳 JSON，不要加說明文字。
"""

        response = model_vision.generate_content([
            prompt,
            {
                "mime_type": image.content_type or "image/jpeg",
                "data": img_bytes
            }
        ])

        raw_text = response.text

        match = re.search(r"\[.*\]", raw_text, re.S)
        if not match:
            raise ValueError(f"Gemini 回傳非 JSON：{raw_text}")

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

