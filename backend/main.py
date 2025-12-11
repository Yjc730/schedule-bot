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
# Memory（圖片 + 對話）
# =========================

"""
chat_memory 結構：

[
  { "role": "user", "content": "問題" },
  { "role": "assistant", "content": "回答" },
  {
    "role": "image",
    "content": "使用者上傳圖片時的文字描述",
    "image_b64": "...",
    "mime": "image/jpeg",
    "summary": "圖片摘要",
    "type": "calendar/table/document/etc"
  }
]
"""

chat_memory: List[Dict[str, Union[str, Dict]]] = []


# =========================
# 圖片分類 Prompt
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


# =========================
# 圖片摘要 Prompt
# =========================
IMAGE_SUMMARY_PROMPT = """
請將圖片的主要內容摘要成 200 字以內。
請包含：

- 圖中主要物件或文字
- 若是文件，摘要段落內容
- 若是表格，摘要欄位與重點資料
- 若是行事曆，摘要有哪些事件（不要逐格描述）
- 若是照片，描述主要場景

禁止：
- 不要回答使用者問題
- 不要推測意圖
- 不要對圖片內容進行延伸分析
"""


# =========================
# 多模態圖片問答 Prompt
# =========================
def build_image_answer_prompt(img_type: str) -> str:
    base = """
你是一個「圖片理解助理」。請根據使用者的問題，
只回答與問題直接相關的資訊。請使用繁體中文。
"""

    if img_type == "calendar":
        base += """
⚠️ 規則：
1. 僅回答問題涉及的日期或時段。
2. 不要列整個行事曆。
3. 若找不到精準答案，請回答「行事曆中沒有找到這個資訊」。
"""
    elif img_type == "table":
        base += """
⚠️ 規則：
1. 根據表格摘要回答欄位或資料。
2. 不要輸出整張表格。
"""
    elif img_type == "document":
        base += """
⚠️ 規則：
1. 根據摘要中的段落內容回答。
2. 回答務必簡潔。
"""
    else:
        base += """
⚠️ 規則：
1. 僅針對摘要內容回答。
2. 若摘要中沒有提到該資訊，請說「我在圖片摘要中沒有看到相關資訊」。
"""

    return base


# =========================
# Health Check
# =========================
@app.get("/")
async def root():
    return {"status": "ok", "message": "Gemini multimodal assistant running"}


# =========================
# Chat API
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(
    message: str = Form(""),
    image: Optional[UploadFile] = File(None)
):
    if not client:
        return ChatResponse(reply="❌ 後端尚未設定 GEMINI_API_KEY")

    # =========================
    # Case 1：圖片 + 問題
    # =========================
    if image is not None:
        img_bytes = await image.read()
        img_b64 = base64.b64encode(img_bytes).decode()
        mime = image.content_type or "image/jpeg"

        # Step 1：圖片類型判斷
        classify_res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                IMAGE_CLASSIFY_PROMPT,
                types.Part.from_bytes(img_bytes, mime)
            ]
        )
        img_type = classify_res.text.strip().lower()

        # Step 2：圖片摘要
        summary_res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                IMAGE_SUMMARY_PROMPT,
                types.Part.from_bytes(img_bytes, mime)
            ]
        )
        img_summary = summary_res.text.strip()

        # Step 3：存入記憶
        chat_memory.append({
            "role": "image",
            "content": message or "[使用者上傳圖片]",
            "image_b64": img_b64,
            "mime": mime,
            "summary": img_summary,
            "type": img_type
        })

        # Step 4：如果有提問 → 回答
        if message:
            prompt = build_image_answer_prompt(img_type)

            convo = [prompt]

            # 帶入最近的圖片摘要
            for m in reversed(chat_memory):
                if m["role"] == "image":
                    convo.append(f"【圖片摘要】：{m['summary']}")
                    break

            convo.append(f"使用者問題：{message}")

            ans = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=convo
            )

            reply = ans.text.strip()
            chat_memory.append({"role": "assistant", "content": reply})
            return ChatResponse(reply=reply)

        # 如果沒問問題 → 回傳摘要
        return ChatResponse(reply=f"已收到圖片，我為你整理的摘要如下：\n\n{img_summary}")

    # =========================
    # Case 2：純文字聊天
    # =========================
    if not message:
        return ChatResponse(reply="請輸入問題或上傳圖片喔！")

    chat_memory.append({"role": "user", "content": message})

    # 組合對話上下文
    convo = ["你是一個自然、口語化、但回答有重點的 AI 助理，請使用繁體中文。"]

    for m in chat_memory[-10:]:
        if m["role"] == "user":
            convo.append(f"使用者：{m['content']}")
        elif m["role"] == "assistant":
            convo.append(f"助理：{m['content']}")
        elif m["role"] == "image":
            convo.append(f"（先前的圖片摘要）：{m['summary']}")

    convo.append(f"使用者：{message}")

    ans = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=convo
    )

    reply = ans.text.strip()
    chat_memory.append({"role": "assistant", "content": reply})
    return ChatResponse(reply=reply)
