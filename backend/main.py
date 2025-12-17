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
# Response Model
# =========================
class ChatResponse(BaseModel):
    reply: str


# =========================
# Conversation Memory
# =========================
chat_memory: List[Dict[str, Union[str, Dict]]] = []


# =========================
# Prompts
# =========================

IMAGE_CLASSIFY_PROMPT = """
你是一個圖片類型分類助手。
請判斷這張圖片屬於以下哪一種：
- 行事曆（calendar）
- 表格（table）
- 文件（document）
- 手寫筆記（handwritten）
- UI 介面（ui）
- 地圖（map）
- 生活照片（photo）
- 其他（other）

只輸出一個類型字串（例如：calendar）
"""

IMAGE_SUMMARY_PROMPT = """
請將圖片或 PDF 頁面的主要內容摘要成 200 字以內。
請包含：
- 主要物件或文字
- 若是文件，摘要段落內容
- 若是表格，摘要欄位與重點資料
- 若是行事曆，摘要有哪些事件（不要逐格描述）
- 若是照片，描述主要場景
"""


# 回答圖片提問的規則
def build_image_answer_prompt(img_type: str) -> str:
    base = """
你是一個「圖片理解助理」。請根據使用者的問題，
只回答與問題直接相關的資訊。請使用繁體中文。
如果你需要額外知識，可以自行推理（使用模型的內建搜尋與推理能力）。
"""

    if img_type == "calendar":
        base += """
⚠️ 規則：
1. 僅回答與日期、節日、事件相關的資訊。
2. 不要逐格念行事曆。
3. 若圖片沒有相關資訊，請回答「行事曆中沒有找到這個資訊」。
"""
    elif img_type == "table":
        base += "請根據摘要回答表格內容，不要重製整張表。"
    elif img_type == "document":
        base += "請根據文件摘要回答，不要過度延伸。"

    return base


# =========================
# Health Check
# =========================
@app.get("/")
def home():
    return {"status": "ok", "message": "AI assistant running"}


# =========================
# Main Chat API
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(
    message: str = Form(""),
    image: Optional[UploadFile] = File(None),
):

    if not client:
        return ChatResponse(reply="❌ server 尚未設定 GEMINI_API_KEY")

    # --------------------------------------
    # CASE 1：有上傳圖片 / PDF
    # --------------------------------------
    if image is not None:

        file_bytes = await image.read()
        mime = image.content_type or "application/octet-stream"

        # 轉 base64
        file_b64 = base64.b64encode(file_bytes).decode()

        # Step 1：分類
        classify = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                IMAGE_CLASSIFY_PROMPT,
                types.Part.from_bytes(file_bytes, mime)
            ]
        )
        img_type = classify.text.strip().lower()

        # Step 2：摘要（PDF 也用相同流程）
        summary = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                IMAGE_SUMMARY_PROMPT,
                types.Part.from_bytes(file_bytes, mime)
            ]
        )
        img_summary = summary.text.strip()

        # Step 3：記錄到 memory
        chat_memory.append({
            "role": "file",
            "filename": image.filename,
            "mime": mime,
            "content": message or "[使用者上傳檔案]",
            "file_b64": file_b64,
            "summary": img_summary,
            "type": img_type
        })

        # Step 4：如果有提問 → 回答
        if message:
            prompt = build_image_answer_prompt(img_type)

            convo = [prompt]

            # 加入最近一份檔案摘要
            for m in reversed(chat_memory):
                if m["role"] == "file":
                    convo.append(f"【檔案摘要】：{m['summary']}")
                    break

            convo.append(f"使用者問題：{message}")

            ans = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=convo
            )

            reply = ans.text.strip()
            chat_memory.append({"role": "assistant", "content": reply})

            return ChatResponse(reply=reply)

        # 如果沒有提問 → 回傳摘要
        return ChatResponse(reply=f"已收到檔案（{image.filename}），以下是摘要：\n\n{img_summary}")

    # --------------------------------------
    # CASE 2：純文字聊天（加入模型自動搜尋能力）
    # --------------------------------------
    if not message:
        return ChatResponse(reply="請輸入訊息或上傳圖片喔！")

    chat_memory.append({"role": "user", "content": message})

    # 對話上下文（保留最近 10 則 + 檔案摘要）
    convo = [
        """
你是一個口語化但專業的 AI 助理。
你可以：
- 使用對話記憶推理
- 使用模型內建搜尋能力自動查找資訊（不需要使用者要求）
- 也可以引用使用者上傳的圖片 / PDF 摘要

請以繁體中文回答。
"""
    ]

    for m in chat_memory[-10:]:
        if m["role"] == "user":
            convo.append(f"使用者：{m['content']}")
        elif m["role"] == "assistant":
            convo.append(f"助理：{m['content']}")
        elif m["role"] == "file":
            convo.append(f"（之前的檔案摘要）：{m['summary']}")

    convo.append(f"使用者：{message}")

    # Gemini 2.5：具備自動搜尋推理能力
    ans = client.models.generate_content(
        model="gemini-2.5",
        contents=convo
    )

    reply = ans.text.strip()
    chat_memory.append({"role": "assistant", "content": reply})

    return ChatResponse(reply=reply)
