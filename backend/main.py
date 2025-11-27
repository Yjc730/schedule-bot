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
# Gemini API Key（從 Railway 讀）
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
# ✅ 聊天上下文記憶
# =========================
chat_memory: List[dict] = []

# =========================
# Root（健康檢查）
# =========================
@app.get("/")
async def root():
    return {"status": "ok", "message": "Gemini AI API Running"}

# =========================
# ✅ 聊天（有上下文 + 像人）
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        chat_memory.append({"role": "user", "content": req.message})

        system_prompt = {
            "role": "system",
            "content": "你是一個溫暖、自然、會用繁體中文聊天的 AI 助手，說話像真人。"
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
        return ChatResponse(reply=f"❌ Gemini 聊天錯誤：{str(e)}")

# =========================
# ✅ 圖片辨識（通用版本：這是什麼圖）
# =========================
@app.post("/parse-schedule-image", response_model=ParseScheduleResponse)
async def parse_schedule_image(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()

        prompt = """
請用「繁體中文」詳細描述這張圖片的內容，
如果圖片中有物品、人物、動作、場景請一併說明，
不要輸出 JSON，只要自然語言說明。
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

        description = response.text.strip()

        # ✅ 包裝成你前端能吃的格式
        return ParseScheduleResponse(events=[
            Event(
                title="圖片內容",
                date="",
                start_time="",
                end_time="",
                location="",
                notes=description,
                raw_text=None,
                source="image"
            )
        ])

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
