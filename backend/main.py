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
    status: str
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
# Root
# =========================
@app.get("/")
async def root():
    return {"status": "ok", "message": "Gemini AI API Running"}

# =========================
# ✅ 一般聊天（助理模式）
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        chat_memory.append({"role": "user", "content": req.message})

        system_prompt = {
            "role": "system",
            "content": """
你是一個溫暖、自然、會用繁體中文聊天的 AI 助手。
可以正常聊天、解釋事情、回答問題。
如果使用者是問圖片解析的內容，你不要亂猜，只根據已解析資料回覆。
"""
        }

        messages = [system_prompt] + chat_memory[-10:]

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[m["content"] for m in messages],
        )

        reply = response.text.strip()
        chat_memory.append({"role": "assistant", "content": reply})

        return ChatResponse(reply=reply)

    except Exception as e:
        return ChatResponse(reply=f"❌ Gemini 聊天錯誤：{str(e)}")

# =========================
# ✅ 行事曆圖片解析（只抓「可用的行程事件」）
# =========================
@app.post("/parse-schedule-image", response_model=ParseScheduleResponse)
async def parse_schedule_image(image: UploadFile = File(...)):
    try:
        img_bytes = await image.read()

        prompt = """
請從這張行事曆圖片中，只擷取「實際有行程的事件」，並輸出為 JSON 陣列（不要說明文字）：

格式如下：
[
  {
    "title": "暫定 / 忙碌 / 會議 / 上課 / 約會 / 工作 / 其他",
    "date": "YYYY-MM-DD",
    "start_time": "HH:MM",
    "end_time": "",
    "status": "暫定 / 忙碌 / 已確定 / 空閒 / 其他",
    "location": "",
    "notes": "",
    "raw_text": "圖片上原始文字",
    "source": "image"
  }
]

⚠️ 規則：
1. 只輸出「看得到具體時間」的行程
2. 不要輸出整個月份介紹
3. 不要輸出 UI 版面描述
4. 不要輸出無日期的內容
5. 僅輸出 JSON 陣列本體
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

        raw_text = response.text.strip()

        # ✅ 強制抽出 JSON 陣列
        match = re.search(r"\[\s*{.*?}\s*\]", raw_text, re.S)
        if not match:
            raise ValueError("沒有解析到有效的事件 JSON")

        events = json.loads(match.group(0))
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

# =========================
# ✅ 「單日行程整理」（給你前端顯示用）
# =========================
@app.post("/format-day-schedule", response_model=ChatResponse)
async def format_day_schedule(req: ParseScheduleResponse):
    try:
        if not req.events:
            return ChatResponse(reply="⚠️ 這一天沒有行程")

        date = req.events[0].date
        lines = [f"📅 {date} 行程："]

        for e in req.events:
            time = e.start_time or "未知時間"
            status = e.status or e.title or "行程"
            lines.append(f"• {time} {status}")

        return ChatResponse(reply="\n".join(lines))

    except Exception as e:
        return ChatResponse(reply=f"格式化失敗：{str(e)}")
