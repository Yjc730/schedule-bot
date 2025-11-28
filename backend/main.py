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
# Gemini API Key
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
    status: Optional[str] = ""
    location: Optional[str] = ""
    notes: Optional[str] = ""
    raw_text: Optional[str] = None
    source: Optional[str] = "image"

class ParseScheduleResponse(BaseModel):
    events: List[Event]

# =========================
# ✅ 全域記憶
# =========================
chat_memory: List[dict] = []
image_events_cache: List[dict] = []   # ✅ 存圖片解析結果

# =========================
# 健康檢查
# =========================
@app.get("/")
async def root():
    return {"status": "ok", "message": "Gemini AI API Running"}

# =========================
# ✅ 聊天（像助理 + 可問行程）
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        user_text = req.message.strip()

        # ✅ 如果是在問某一天的行程
        date_match = re.search(r"(\d{1,2})\s*[日號]", user_text)

        if date_match and image_events_cache:
            day = date_match.group(1).zfill(2)
            result = []

            for e in image_events_cache:
                if e["date"].endswith(f"-{day}"):
                    result.append(
                        f"• {e['start_time']} {e['title']}"
                    )

            if result:
                reply = f"📅 {int(day)} 日行程：\n" + "\n".join(result)
                return ChatResponse(reply=reply)
            else:
                return ChatResponse(reply=f"📅 {int(day)} 日沒有行程")

        # ✅ 一般聊天模式
        system_prompt = {
            "role": "system",
            "content": "你是一個溫暖、自然、會用繁體中文聊天的 AI 助手。"
        }

        chat_memory.append({"role": "user", "content": user_text})
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
# ✅ 圖片解析（真正結構化版本）
# =========================
@app.post("/parse-schedule-image", response_model=ParseScheduleResponse)
async def parse_schedule_image(image: UploadFile = File(...)):
    global image_events_cache

    try:
        img_bytes = await image.read()

        prompt = """
你現在是行事曆辨識系統。
請從圖片中「只擷取真正的行程」，並輸出為 JSON 陣列：

欄位格式：
[{
  "title": "暫定 / 忙碌 / 會議 / 課程",
  "date": "YYYY-MM-DD",
  "start_time": "HH:MM",
  "end_time": "",
  "status": "",
  "location": "",
  "notes": "",
  "raw_text": "",
  "source": "image"
}]

❗規則：
1️⃣ 只能回傳 JSON
2️⃣ 不要任何說明文字
3️⃣ 不要整月
4️⃣ 只回傳「真正有標記事件的格子」
5️⃣ 如果圖片沒有行程，回傳空陣列 []
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=img_bytes,
                    mime_type=image.content_type or "image/jpeg"
                ),
                prompt,
            ],
        )

        raw = response.text.strip()

        # ✅ 強制萃取 JSON
        match = re.search(r"\[.*\]", raw, re.S)
        if not match:
            raise ValueError(f"非 JSON 回傳：{raw}")

        events = json.loads(match.group(0))

        # ✅ 快取全月行程（給聊天查詢）
        image_events_cache = events

        return ParseScheduleResponse(events=events)

    except Exception as e:
        return ParseScheduleResponse(events=[
            Event(
                title="圖片解析失敗",
                date="",
                start_time="",
                end_time="",
                status="error",
                location="",
                notes=str(e),
                raw_text=None,
                source="image"
            )
        ])
