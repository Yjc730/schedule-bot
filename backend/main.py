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
class ChatResponse(BaseModel):
    reply: str

# =========================
# 健康檢查
# =========================
@app.get("/")
async def root():
    return {"status": "ok"}

# =========================
# ✅【唯一入口】文字 + 圖片 合併處理
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(
    message: str = Form(...),
    image: UploadFile = File(None)
):
    try:
        img_part = None

        # ✅ 如果有圖片 → 加入 vision
        if image:
            img_bytes = await image.read()
            img_part = types.Part.from_bytes(
                data=img_bytes,
                mime_type=image.content_type or "image/jpeg"
            )

        # ✅ 嚴格限制回覆格式（避免他亂講整個月）
        prompt = f"""
你是「行事曆 AI 助理」。
規則極度嚴格：

1️⃣ 若使用者有指定「某一天」：
只回該日的行程
格式必須為：

📅 31 日行程：
• 09:30 暫定
• 10:00 忙碌

2️⃣ 若圖片中只有節日：
只回答節日結果，例如：
「除夕是 2023-01-21。」

3️⃣ 禁止列出整個月份
4️⃣ 禁止輸出 JSON
5️⃣ 禁止解釋過程
6️⃣ 只能用繁體中文

使用者問題：
{message}
"""

        contents = [prompt]
        if img_part:
            contents = [img_part, prompt]

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents
        )

        reply = response.text.strip()
        return ChatResponse(reply=reply)

    except Exception as e:
        return ChatResponse(reply=f"❌ 系統錯誤：{str(e)}")
