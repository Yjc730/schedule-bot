import os
from typing import List, Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.genai as genai
from google.genai import types

# =========================
# Gemini API Key
# =========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# =========================
# FastAPI
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 你前端在哪都可以打進來
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Models
# =========================
class ChatResponse(BaseModel):
    reply: str

# 簡單聊天記憶（只存最近幾句）
chat_memory: List[Dict[str, str]] = []

# =========================
# Health Check
# =========================
@app.get("/")
async def root():
    return {"status": "ok", "message": "Gemini calendar + chat bot running"}

# =========================
# 單一入口：/chat
# - 同時支援：純文字聊天
# - 以及：文字 + 行事曆圖片 問問題
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(
    message: str = Form(""),
    image: Optional[UploadFile] = File(None),
):
    if not client:
        return ChatResponse(reply="❌ 後端尚未設定 GEMINI_API_KEY")

    try:
        # ---------------------------------
        # 情境 1：有上傳圖片（多半是行事曆）
        # ---------------------------------
        if image is not None:
            img_bytes = await image.read()

            calendar_prompt = """
你是一個「行事曆圖片助理」，請嚴格遵守下面規則：

1. 使用者會同時：
   - 上傳一張行事曆 / 月曆 / 行程表的圖片
   - 輸入一段問題（例如：「除夕是哪一天？」「1/31 上午 9:30 是什麼狀態？」）

2. 你的回答 **一定要非常精簡**，只針對問題問到的日期 / 時間回答。
   - 問：「除夕是哪一天？」 ➜ 回：「除夕是 1 月 21 日。」
   - 問：「星期二 31 日 上午 9:30 是什麼？」 ➜ 回：「星期二 31 日 上午 9:30 顯示『暫定』。」

3. **禁止** 列出整個月份或所有行程。
   - 不要輸出表格
   - 不要輸出 JSON
   - 不要解釋圖片的每一格

4. 如果圖片裡找不到精確答案，就簡短說「我在行事曆裡沒有找到這個資訊」，並請使用者再確認。

5. 如果使用者沒有輸入任何文字問題，只上傳行事曆，
   就用 1～2 句話，簡單說明這張行事曆的大致內容（例如有哪些重要節日或提醒）。

請依照上面規則，用繁體中文，口語、簡潔地回答。
"""

            contents = [
                calendar_prompt,
                types.Part.from_bytes(
                    data=img_bytes,
                    mime_type=image.content_type or "image/jpeg",
                )
            ]

            if message:
                contents.append(f"使用者問題：{message}")

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents
            )

            reply = (response.text or "").strip()
            if not reply:
                reply = "我好像沒在行事曆裡找到明確答案，可以再描述一次你的問題嗎？"

            # 把文字對話存進記憶（但不存圖片）
            if message:
                chat_memory.append({"role": "user", "content": message})
                chat_memory.append({"role": "assistant", "content": reply})

            return ChatResponse(reply=reply)

        # ---------------------------------
        # 情境 2：純文字聊天
        # ---------------------------------
        if not message:
            return ChatResponse(reply="可以先輸入一段文字，或搭配行事曆圖片一起詢問喔！")

        system_prompt = (
            "你是一個溫暖、自然、會用繁體中文回答的 AI 助理，"
            "講話像真人一樣口語，但回答要有重點、不要太長。"
        )

        # 紀錄使用者訊息
        chat_memory.append({"role": "user", "content": message})
        # 只帶最近 10 則對話
        recent = chat_memory[-10:]

        convo = [system_prompt] + [
            f"{m['role']}：{m['content']}" for m in recent
        ]

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=convo
        )
        reply = (response.text or "").strip() or "我想了一下，好像沒有得到內容，可以再問一次嗎？"

        chat_memory.append({"role": "assistant", "content": reply})
        return ChatResponse(reply=reply)

    except Exception as e:
        # 不要讓 exception 直接炸掉服務
        return ChatResponse(reply=f"❌ 伺服器錯誤：{str(e)}")
