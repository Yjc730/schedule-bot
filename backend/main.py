import os
import json
import re
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
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

# =========================
# 健康檢查
# =========================
@app.get("/")
async def root():
    return {"status": "ok"}

# =========================
# ✅【唯一入口：文字 + 圖片 + 行事曆 + 一般聊天】
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(
    message: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    try:
        contents = []

        # ✅ 有圖片就丟進去
        if image:
            img_bytes = await image.read()
            contents.append(
                types.Part.from_bytes(
                    data=img_bytes,
                    mime_type=image.content_type or "image/jpeg"
                )
            )

        # ✅ 系統提示（這段是關鍵）
        system_prompt = f"""
你是一個「行事曆 + 一般聊天 AI 助理」。
規則：
1️⃣ 如果使用者只是聊天 → 正常回答
2️⃣ 如果使用者有上傳圖片 → 視為「行事曆圖片」
3️⃣ 如果使用者的文字有指定日期（例如：除夕、31日、星期二）：
   ✅ 只回那一天
   ✅ 禁止回整個月份
   ✅ 禁止輸出 JSON
   ✅ 只用這個格式：

📅 31 日行程：
• 09:30 暫定
• 10:00 忙碌

4️⃣ 如果圖片中該天沒有事件 → 明確說「該日沒有行程」
5️⃣ 嚴禁解釋你怎麼解析
"""

        user_text = message or "請協助分析圖片中的行事曆"

        contents.append(system_prompt)
        contents.append(user_text)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
        )

        reply = response.text.strip()

        # ✅ 防止模型亂噴 JSON
        if reply.startswith("{") or reply.startswith("["):
            reply = "⚠️ 目前只能顯示該日摘要，請重新提問。"

        return ChatResponse(reply=reply)

    except Exception as e:
        return ChatResponse(reply=f"❌ 發生錯誤：{str(e)}")
