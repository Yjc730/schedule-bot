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
# 全域記憶
# =========================
chat_memory: List[dict] = []
# 👇 圖片解析後的所有行程都塞在這裡
image_events_cache: List[dict] = []

# =========================
# 健康檢查
# =========================
@app.get("/")
async def root():
    return {"status": "ok", "message": "Gemini AI API Running"}

# =========================
# ✅ 聊天（支援：助理聊天 + 問某天 / 問某節日）
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        user_text = req.message.strip()
        has_events = len(image_events_cache) > 0

        # ---------- 1️⃣ 問「某一天的行程」：例如 31 日行程 ----------
        day_match = re.search(r"(\d{1,2})\s*[日号號]", user_text)
        if day_match and has_events and ("行程" in user_text or "schedule" in user_text):
            day = day_match.group(1).zfill(2)  # 31 -> "31"
            items = []

            for e in image_events_cache:
                date = e.get("date", "")
                if date.endswith(f"-{day}"):
                    start = e.get("start_time", "")
                    title = e.get("title", "")
                    if start or title:
                        items.append(f"• {start} {title}".strip())

            if items:
                reply = f"📅 {int(day)} 日行程：\n" + "\n".join(items)
            else:
                reply = f"📅 {int(day)} 日沒有找到行程喔～"

            return ChatResponse(reply=reply)

        # ---------- 2️⃣ 問「某個節日是哪一天」：例如 除夕是哪一天 ----------
        # 從目前的 events 裡面抓出可能的「關鍵字」(title/raw_text)
        if has_events and ("哪一天" in user_text or "哪天" in user_text or "幾號" in user_text):
            # 把 user 問的文字拿去對 events 的 title / raw_text 做包含搜尋
            keyword = None
            for e in image_events_cache:
                for field in ["title", "raw_text"]:
                    val = (e.get(field) or "").strip()
                    if val and val in user_text:
                        keyword = val
                        break
                if keyword:
                    break

            if keyword:
                matched_dates = set()
                for e in image_events_cache:
                    title = (e.get("title") or "")
                    raw = (e.get("raw_text") or "")
                    if keyword in title or keyword in raw:
                        if e.get("date"):
                            matched_dates.add(e["date"])

                if matched_dates:
                    dates_sorted = sorted(matched_dates)
                    if len(dates_sorted) == 1:
                        reply = f"📅「{keyword}」是在 {dates_sorted[0]}。"
                    else:
                        reply = "📅 找到多個日期：\n" + "\n".join(f"• {d}" for d in dates_sorted)
                    return ChatResponse(reply=reply)

        # ---------- 3️⃣ 一般聊天：當作暖暖的中文 AI 助理 ----------
        system_prompt = {
            "role": "system",
            "content": "你是一個溫暖、自然、會用繁體中文聊天的 AI 助手，語氣像真人、輕鬆好聊。"
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
# ✅ 圖片解析（行事曆 → 乾淨 JSON + 快取）
# =========================
@app.post("/parse-schedule-image", response_model=ParseScheduleResponse)
async def parse_schedule_image(image: UploadFile = File(...)):
    """
    這個 API 的角色很單純：
    1. 把行事曆圖片解析成 events JSON
    2. 存進 image_events_cache，給 /chat 後續查詢用
    """
    global image_events_cache

    try:
        img_bytes = await image.read()

        prompt = """
你現在是一個「行事曆辨識系統」。
請從圖片中擷取所有「有內容的格子」，輸出成 JSON 陣列：

[{
  "title": "節日 / 行程名稱（例如：除夕、春節、會議、暫定、忙碌）",
  "date": "YYYY-MM-DD",
  "start_time": "",       // 有時間就填 HH:MM，沒有就留空字串
  "end_time": "",
  "status": "",           // 忙碌 / 暫定 / 放假 ... 沒有就空字串
  "location": "",
  "notes": "",
  "raw_text": "該格子原始文字",
  "source": "image"
}]

⚠️ 規則：
1. 只能輸出 JSON 陣列，不要任何說明文字
2. 如果沒有任何事件，回傳 []
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

        # 從回傳文字中抓出 JSON 陣列
        match = re.search(r"\[.*\]", raw, re.S)
        if not match:
            raise ValueError(f"非 JSON 回傳：{raw}")

        events = json.loads(match.group(0))

        # ✅ 把 events 存起來，給 /chat 用
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
