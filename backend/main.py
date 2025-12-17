import os
import base64
from typing import List, Dict, Optional, Union
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.genai as genai
from google.genai import types

# =========================
# Gemini API
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

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
# Memory（圖片 / PDF / 對話）
# =========================
chat_memory: List[Dict[str, Union[str, Dict]]] = []

# =========================
# Prompts
# =========================
IMAGE_CLASSIFY_PROMPT = """
你是一個圖片/文件類型分類助手。
請判斷這份檔案屬於以下哪一種：
- 行事曆（calendar）
- 表格（table）
- 文件（document）
- 手寫筆記（handwritten）
- UI 介面（ui）
- 地圖（map）
- 生活照片（photo）
- PDF（pdf）
- 其他（other）
只輸出類型字串（例如：calendar）
"""

IMAGE_SUMMARY_PROMPT = """
請將圖片或 PDF 的主要內容摘要成 200 字以內。
請包含：
- 主要物件或文字
- 若是文件，摘要段落內容
- 若是表格，摘要欄位與重點資料
- 若是行事曆，摘要重大事件（不要逐格描述）
禁止：
- 不要加入你自己的推測
- 不要回答使用者問題（僅摘要）
"""

def build_answer_prompt(file_type: str) -> str:
    base = """
你是一個「圖片/文件理解助理」。請根據使用者問題，
只回答與問題直接相關的內容。請使用繁體中文。
必要時，你可以使用模型的自動搜尋能力補充知識。
"""

    if file_type == "calendar":
        base += """
⚠️ 規則：
1. 僅回答與日期/事件相關。
2. 若找不到資訊，回答「行事曆中沒有找到這個資訊」。
"""
    elif file_type == "table":
        base += "根據表格摘要回答內容，不要重製完整表格。"
    elif file_type == "document":
        base += "根據文件摘要回答問題，不要過度延伸。"

    return base


# =========================
# Health Check
# =========================
@app.get("/")
def home():
    return {"status": "ok", "message": "AI assistant running"}


# =========================
# Chat API
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(
    message: str = Form(""),
    image: Optional[UploadFile] = File(None),
):
    if not client:
        return ChatResponse(reply="❌ 尚未設定 GEMINI_API_KEY")

    # ======================================================
    # CASE 1 — 上傳檔案（圖片 / PDF）
    # ======================================================
    if image is not None:
        file_bytes = await image.read()
        mime = image.content_type or "application/octet-stream"

        # 供記憶使用（不傳給 Gemini）
        file_b64 = base64.b64encode(file_bytes).decode()

        # ---- Step 1：檔案分類 ----
        classify = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                IMAGE_CLASSIFY_PROMPT,
                types.Part.from_bytes(file_bytes, mime_type=mime)
            ]
        )
        file_type = classify.text.strip().lower()

        # ---- Step 2：檔案摘要 ----
        summary = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                IMAGE_SUMMARY_PROMPT,
                types.Part.from_bytes(file_bytes, mime_type=mime)
            ]
        )
        file_summary = summary.text.strip()

        # ---- Step 3：寫入記憶 ----
        chat_memory.append({
            "role": "file",
            "filename": image.filename,
            "mime": mime,
            "summary": file_summary,
            "type": file_type
        })

        # ---- Step 4：若使用者同時有提問 ----
        if message:
            prompt = build_answer_prompt(file_type)

            convo = [prompt]
            convo.append(f"【檔案摘要】：{file_summary}")
            convo.append(f"使用者問題：{message}")

            ans = client.models.generate_content(
                model="gemini-2.5",
                contents=convo
            )
            reply = ans.text.strip()

            chat_memory.append({"role": "assistant", "content": reply})
            return ChatResponse(reply=reply)

        # ---- 若沒有提問，回傳摘要 ----
        return ChatResponse(
            reply=f"已收到檔案（{image.filename}），以下是摘要：\n\n{file_summary}"
        )

    # ======================================================
    # CASE 2 — 純文字聊天（Gemini 2.5 自動搜尋）
    # ======================================================
    if not message:
        return ChatResponse(reply="請輸入訊息或上傳圖片/文件喔！")

    chat_memory.append({"role": "user", "content": message})

    # 建立對話上下文
    convo = [
        """
你是一個專業、口語化、能自動搜尋最新資訊的 AI 助理。
你可以：
- 使用上下文記憶推理
- 使用模型自動搜尋能力查找答案
- 結合圖片/PDF 摘要回應
回答請使用繁體中文。
"""
    ]

    # 帶入最近 10 則對話
    for m in chat_memory[-10:]:
        if m["role"] == "user":
            convo.append(f"使用者：{m['content']}")
        elif m["role"] == "assistant":
            convo.append(f"助理：{m['content']}")
        elif m["role"] == "file":
            convo.append(f"（之前檔案摘要）：{m['summary']}")

    convo.append(f"使用者：{message}")

    # 使用 Gemini 2.5 → 自動搜尋
    ans = client.models.generate_content(
        model="gemini-2.5",
        contents=convo
    )
    reply = ans.text.strip()

    chat_memory.append({"role": "assistant", "content": reply})
    return ChatResponse(reply=reply)
