from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()

# CORS：本機開 index.html 也能呼叫 http://localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 正式上線可以改成你的網域
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    reference_date: Optional[str] = None  # YYYY-MM-DD


class Event(BaseModel):
    title: str
    date: str        # YYYY-MM-DD
    start_time: str  # HH:MM
    end_time: str    # HH:MM
    location: Optional[str] = ""
    notes: Optional[str] = ""
    raw_text: Optional[str] = None
    source: Optional[str] = "text"


class ParseScheduleResponse(BaseModel):
    events: List[Event]


@app.get("/")
async def root():
    return {"status": "ok", "message": "Schedule Bot API"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # TODO: 這裡之後改成呼叫真正的 LLM API
    reply_text = f"（假回覆）你剛剛說：{req.message}"
    return ChatResponse(reply=reply_text)


@app.post("/parse-schedule-text", response_model=ParseScheduleResponse)
async def parse_schedule_text(req: ParseTextRequest):
    """
    目前先用非常簡單的假邏輯：
    - 不真的解析文字
    - 只回傳一筆固定的 Event
    之後可以在這裡串接 LLM，讓它輸出 JSON，再 parse 成 Event。
    """
    dummy_event = Event(
        title="打掃",
        date="2025-11-25",
        start_time="13:00",
        end_time="14:00",
        location="家裡",
        notes="這是示範用假資料，之後改成真正解析結果",
        raw_text=req.text,
        source="text",
    )
    return ParseScheduleResponse(events=[dummy_event])


@app.post("/parse-schedule-image", response_model=ParseScheduleResponse)
async def parse_schedule_image(
    image: UploadFile = File(...),
    reference_date: Optional[str] = Form(None),
):
    """
    目前先不真的解析圖片，只回傳一筆假資料。
    之後可以：
    - 讀取 image.file
    - 轉成 base64
    - 丟給 Vision LLM API
    """
    filename = image.filename
    dummy_event = Event(
        title=f"從圖片（{filename}）讀到的假行程",
        date="2025-11-26",
        start_time="10:00",
        end_time="11:00",
        location="不明地點",
        notes="這是示範用假資料，之後改成 Vision LLM 的結果",
        raw_text=None,
        source="image",
    )
    return ParseScheduleResponse(events=[dummy_event])
