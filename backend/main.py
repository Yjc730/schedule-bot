import os
import json
import re
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import google.genai as genai
from google.genai import types

# =========================
# Gemini API
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
class ChatResponse(BaseModel):
    reply: str

class Event(BaseModel):
    title: str
    date: str
    start_time: str
    end_time: str
    status: Optional[str] = ""
    location: Optional[str] = ""
    notes: Optional[str] = ""
    raw_text: Optional[str] = None
    source: Optional[str] = "image"

class ParseScheduleResponse(BaseModel):
    events: List[Event]

# =========================
# 健康檢查
# =========================
@app.get("/")
async def root():
    return {"status": "ok"}

# ✅✅✅ ✅✅✅ ✅✅✅
# ✅【1】圖片 + 文字 同時送的 API
# ✅✅✅ ✅✅✅ ✅✅✅
@app.post("/chat-with-image", response_model=ChatResponse)
async def chat_with_image(
    message: str = Form(...),
    image: UploadFile = File(None)
):
    try:
        contents = []

        # ✅ 有圖片就一起送
        if image:
            img_bytes = await image.read()
            contents.append(
                types.Part.from_bytes(
                    data=img_bytes,
                    mime_type=image.content_type or "image/jpeg",
                )
            )

        # ✅ 強制輸出只回答使用者問題
        prompt = f"""
你是一個「行事曆 + 一般聊天」助理。

【嚴格規則】
1️⃣ 只能回答「使用者問的那一天或那一個事件」
2️⃣ 禁止列出整個月
3️⃣ 禁止補充其他節日
4️⃣ 若圖片中沒有該問題的答案，只回：
   「圖片中沒有找到該資訊」

【輸出格式】
📅 XX 日行程：
• HH:MM 狀態
• HH:MM 狀態

使用者問題：
{message}
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents + [prompt],
        )

        return ChatResponse(reply=response.text.strip())

    except Exception as e:
        return ChatResponse(reply=f"❌ 解析失敗：{str(e)}")

# ✅✅✅ ✅✅✅ ✅✅✅
# ✅【2】純聊天室（沒有圖片）
# ✅✅✅ ✅✅✅ ✅✅✅
@app.post("/chat", response_model=ChatResponse)
async def chat(message: str = Form(...)):
    try:
        prompt = f"""
你是一般聊天 AI 助理，若不是行事曆問題就正常對話。

使用者說：
{message}
"""
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt]
        )
        return ChatResponse(reply=response.text.strip())

    except Exception as e:
        return ChatResponse(reply=f"❌ 錯誤：{str(e)}")
