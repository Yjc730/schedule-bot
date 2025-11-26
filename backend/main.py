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
# ✅ 正確的 Gemini API Key 讀取方式
# ✅ 請在 Railway 設定：
# GEMINI_API_KEY=你的金鑰
# =========================
GEMINI_API_KEY = os.getenv("AIzaSyApby4uGU1rqVKMLG76dkX8nnZ0zFUnd2M")

if not GEMINI_API_KEY:
    raise RuntimeError("❌ 缺少 GEMINI_API_KEY，請到 Railway 設定環境變數")

# ✅ 初始化 Gemini Client（只初始化一次，避免衝突）
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
# Root
# =========================
@app.get("/")
async def root():
    return {"status": "ok", "message": "Gemini AI API Running"}

# =========================
# ✅ ✅ 聊天（Gemini 2.5 Flash）
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=req.message
    )
    return ChatResponse(reply=response.text)

# =========================
# ✅ ✅ ✅ 圖片解析（Gemini 2.5 Flash Vision）
# =========================
@app.post("/parse-schedule-image", response_model=ParseScheduleResponse)
async def parse_schedule_image(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()

        prompt = """
你是一個專業行事曆圖片解析 AI。
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
⚠️ 只回傳 JSON 陣列本體，不要加說明文字。
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
