import os
import json
import google.genai as genai
from google.genai import types
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# =====================
# Gemini
# =====================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# =====================
# FastAPI
# =====================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# Models
# =====================
class ChatResponse(BaseModel):
    reply: str

# =====================
# Health Check
# =====================
@app.get("/")
async def root():
    return {"status": "ok"}

# =====================
# ✅【統一聊天 + 圖片解析 API】
# =====================
@app.post("/chat", response_model=ChatResponse)
async def chat(
    message: Optional[str] = Form(""),
    image: Optional[UploadFile] = File(None)
):
    try:
        parts = []

        # ✅ 圖片存在 → 送入 Gemini Vision
        if image:
            img_bytes = await image.read()
            parts.append(
                types.Part.from_bytes(
                    data=img_bytes,
                    mime_type=image.content_type or "image/jpeg"
                )
            )

        # ✅ 強制限制模型只回答「使用者詢問的那一天」
        system_prompt = """
你是一個「行事曆 + 一般聊天」AI 助理。

規則：
1️⃣ 如果使用者問「某一天的行程」：
→ 你只能輸出該"指定日期"
→ 嚴禁輸出其他日期
→ 嚴禁輸出整個月份
→ 格式必須是：

📅 31 日行程：
• 09:30 暫定
• 10:00 忙碌

2️⃣ 如果使用者只是一般聊天 → 正常回答。

3️⃣ 如果有圖片：
→ 你必須先從圖片讀取行事曆內容再回答問題
→ 只回問題相關的日期
→ 不要輸出 JSON
→ 不要輸出其他節日

使用者問題：
"""
        parts.append(system_prompt + message)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=parts
        )

        return ChatResponse(
            reply=response.text.strip()
        )

    except Exception as e:
        return ChatResponse(reply=f"❌ 發生錯誤：{str(e)}")
