import os
import base64
import json
import re
import requests
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# =========================
# OpenRouter 設定
# =========================
API_KEY = os.getenv("API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TEXT_MODEL = "qwen/qwen-2.5-7b-instruct"
VISION_MODEL = "qwen/qwen-2.5-vl-7b"

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
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    context: Optional[List[ChatMessage]] = None

class ChatResponse(BaseModel):
    reply: str

class ParseTextRequest(BaseModel):
    text: str
    reference_date: Optional[str] = None

class Event(BaseModel):
    title: str
    date: str
    start_time: str
    end_time: str
    location: Optional[str] = ""
    notes: Optional[str] = ""
    raw_text: Optional[str] = None
    source: Optional[str] = "text"

class ParseScheduleResponse(BaseModel):
    events: List[Event]

# =========================
# Root
# =========================
@app.get("/")
async def root():
    return {"status": "ok", "message": "Schedule Bot API"}

# =========================
# ✅ 真・LLM 聊天
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [{"role": "system", "content": "你是一個友善的繁體中文 AI 助手。"}]
    if req.context:
        for msg in req.context:
            messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": req.message})

    payload = {
        "model": TEXT_MODEL,
        "messages": messages,
        "temperature": 0.7,
    }

    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    data = r.json()
    reply_text = data["choices"][0]["message"]["content"]
    return ChatResponse(reply=reply_text)

# =========================
# ✅ 文字 → JSON 行程解析
# =========================
@app.post("/parse-schedule-text", response_model=ParseScheduleResponse)
async def parse_schedule_text(req: ParseTextRequest):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
請把下面的行程描述轉成 JSON 陣列：

[{{
  "title": "",
  "date": "YYYY-MM-DD",
  "start_time": "HH:MM",
  "end_time": "HH:MM",
  "location": "",
  "notes": "",
  "raw_text": "",
  "source": "text"
}}]

行程描述：
{req.text}

⚠️ 只回傳 JSON，不要加說明文字。
"""

    payload = {
        "model": TEXT_MODEL,
        "messages": [
            {"role": "system", "content": "你是專業行程解析 AI。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
    }

    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    data = r.json()
    raw_text = data["choices"][0]["message"]["content"]

    match = re.search(r"\[.*\]", raw_text, re.S)
    events = json.loads(match.group(0))

    return ParseScheduleResponse(events=events)

# =========================
# ✅ ✅ ✅ 圖片 → Vision 真解析（最終穩定版）
# =========================
@app.post("/parse-schedule-image", response_model=ParseScheduleResponse)
async def parse_schedule_image(
    image: UploadFile = File(...),
    reference_date: Optional[str] = Form(None),
):
    if not API_KEY:
        return ParseScheduleResponse(events=[
            Event(
                title="系統錯誤：缺少 API_KEY",
                date="",
                start_time="",
                end_time="",
                location="",
                notes="請在 Railway 設定環境變數 API_KEY",
                raw_text=None,
                source="image"
            )
        ])

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    img_bytes = await image.read()
    b64_img = base64.b64encode(img_bytes).decode("utf-8")

    prompt = """
你是一個專業行事曆圖片解析 AI。
請從圖片中辨識所有行程，並輸出為 JSON 陣列：

[{
  "title": "",
  "date": "YYYY-MM-DD",
  "start_time": "HH:MM",
  "end_time": "HH:MM",
  "location": "",
  "notes": "",
  "raw_text": null,
  "source": "image"
}]

⚠️ 僅回傳 JSON 陣列，不要加說明文字。
"""

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64_img}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.2,
    }

    try:
        r = requests.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=120
        )

        data = r.json()
        raw_text = data["choices"][0]["message"]["content"]

        match = re.search(r"\[.*\]", raw_text, re.S)
        if not match:
            raise ValueError(f"Vision 回傳非 JSON：{raw_text}")

        clean_json = match.group(0)
        events = json.loads(clean_json)

        return ParseScheduleResponse(events=events)

    except Exception as e:
        return ParseScheduleResponse(events=[
            Event(
                title="圖片解析失敗",
                date="",
                start_time="",
                end_time="",
                location="",
                notes=str(e),
                raw_text=None,
                source="image"
            )
        ])
