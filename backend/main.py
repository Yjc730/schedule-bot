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
    allow_origins=["*"],   # 方便前端直接連
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
# 記憶：對話 + 檔案摘要
# =========================
"""
chat_memory 內容長這樣：

[
  { "role": "user", "content": "問題" },
  { "role": "assistant", "content": "回答" },
  {
    "role": "file",
    "content": "使用者上傳檔案時說的話",
    "file_b64": "...",
    "mime": "image/jpeg" / "application/pdf" / ...
    "summary": "Gemini 幫檔案做的摘要",
    "kind": "calendar" / "table" / "document" / ...
  }
]
"""

chat_memory: List[Dict[str, Union[str, Dict]]] = []


# =========================
# 檔案類型分類 Prompt（圖片 / PDF 共用）
# =========================
FILE_CLASSIFY_PROMPT = """
你是一個「檔案／圖片類型分類助手」。
使用者會上傳圖片或 PDF，請判斷檔案的主要類型，只能從下列選一個輸出：

- calendar   ：行事曆或月曆、週曆
- table      ：資料表格
- document   ：一般文件、報告、簡報
- pdf        ：多頁 PDF 資料（非明顯表格或行事曆）
- handwritten：手寫筆記
- ui         ：介面截圖（App、網站）
- map        ：地圖
- photo      ：生活／場景照片
- other      ：其他無法歸類

⚠️ 請「只輸出一個小寫英文單字」，例如：calendar
不要加任何說明文字。
"""

# =========================
# 檔案摘要 Prompt（圖片 / PDF 共用）
# =========================
FILE_SUMMARY_PROMPT = """
你是一個「檔案內容摘要助手」，請使用繁體中文回答。

請根據使用者上傳的圖片或 PDF，把內容整理成 200 字以內的摘要，包含：
- 檔案的大致類型與主題
- 若是行事曆：說明主要月份與幾個重要事件（不用列出全部日期）
- 若是表格：說明欄位名稱，以及幾個關鍵數值或趨勢
- 若是文件：簡單整理主要段落重點
- 若是照片：描述主要場景與人物／物件

禁止：
- 不要幫使用者做推論或建議
- 不要回答還沒被問到的問題
- 不要出現「這是一張圖片／PDF」之類的廢話
"""


# =========================
# 根據檔案類型建立回答 Prompt
# =========================
def build_file_answer_prompt(file_kind: str) -> str:
    base = """
你是一個「圖片／PDF 理解小幫手」，請使用繁體中文，
只能根據下方提供的檔案摘要（還有對話內容）來回答使用者的問題。

若摘要裡沒有相關資訊，就老實說：
「在目前的檔案摘要裡沒有找到這個資訊。」

禁止：
- 不要說自己「看不到圖片」或「無法開啟 PDF」。
- 不要要使用者再上傳一次同一個檔案。
"""

    file_kind = (file_kind or "").lower()

    if file_kind == "calendar":
        base += """
【特別規則（行事曆）】
1. 只回答問題涉及的日期或時段，不要把整個月的所有行程全部列出。
2. 如果使用者問「X 月有哪些節日 / 行程」，可以列出 3～6 個重點就好。
"""
    elif file_kind == "table":
        base += """
【特別規則（表格）】
1. 針對欄位與數值做重點整理，可以說出趨勢或比較。
2. 不要整張表直接唸出來。
"""
    elif file_kind in ("document", "pdf"):
        base += """
【特別規則（文件 / PDF）】
1. 請根據摘要中的內容回答重點，適度整理結構（條列式 OK）。
2. 不要逐字重複原文，而是用自己的話濃縮。
"""
    else:
        base += """
【特別規則（其他類型）】
1. 僅根據摘要中出現的內容回答。
2. 若摘要裡真的完全沒有相關資訊，就說找不到，不要亂猜。
"""

    return base.strip()


# =========================
# Health Check
# =========================
@app.get("/")
async def root():
    return {"status": "ok", "message": "Gemini multimodal assistant running"}


# =========================
# Chat API（文字 + 圖片 + PDF）
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(
    message: str = Form(""),
    image: Optional[UploadFile] = File(None),  # 這裡的 image 其實可以是圖片或 PDF
):
    if not client:
        return ChatResponse(reply="❌ 後端尚未設定 GEMINI_API_KEY")

    # ==============
    # Case 1：有檔案（圖片 or PDF）
    # ==============
    if image is not None:
        file_bytes = await image.read()
        # 判斷 MIME
        mime = image.content_type or "application/octet-stream"
        filename = image.filename or ""
        ext = filename.lower().split(".")[-1] if "." in filename else ""

        # ---- Step 1：類型分類 ----
        try:
            classify_res = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    FILE_CLASSIFY_PROMPT,
                    types.Part.from_bytes(data=file_bytes, mime_type=mime),
                ],
            )
            file_kind = classify_res.text.strip().lower()
        except Exception:
            file_kind = "other"

        # ---- Step 2：內容摘要 ----
        try:
            summary_res = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    FILE_SUMMARY_PROMPT,
                    types.Part.from_bytes(data=file_bytes, mime_type=mime),
                ],
            )
            summary_text = summary_res.text.strip()
        except Exception as e:
            summary_text = f"檔案摘要失敗：{e}"

        # ---- Step 3：存到記憶 ----
        file_entry: Dict[str, Union[str, Dict]] = {
            "role": "file",
            "content": message or "[使用者上傳檔案]",
            "file_b64": base64.b64encode(file_bytes).decode(),
            "mime": mime,
            "summary": summary_text,
            "kind": file_kind,
            "name": filename,
        }
        chat_memory.append(file_entry)

        # 控制記憶長度
        if len(chat_memory) > 50:
            chat_memory.pop(0)

        # ---- Step 4：如果同時有提問，直接回答 ----
        if message:
            prompt = build_file_answer_prompt(file_kind)

            # 重新把檔案丟進去，讓模型真的「看得到」它
            file_part = types.Part.from_bytes(data=file_bytes, mime_type=mime)

            contents = [
                file_part,
                prompt,
                f"這是系統為該檔案產生的摘要：\n{summary_text}",
                f"使用者現在的提問：{message}",
            ]

            try:
                ans = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                )
                reply = ans.text.strip()
            except Exception as e:
                reply = f"檔案相關回答時發生錯誤：{e}"

            chat_memory.append({"role": "assistant", "content": reply})
            if len(chat_memory) > 50:
                chat_memory.pop(0)

            return ChatResponse(reply=reply)

        # 如果沒有問題，只是單純上傳檔案，就回傳摘要
        reply = f"✅ 已收到檔案 **{filename or ''}** ，以下是幫你整理的重點摘要：\n\n{summary_text}"
        chat_memory.append({"role": "assistant", "content": reply})
        if len(chat_memory) > 50:
            chat_memory.pop(0)

        return ChatResponse(reply=reply)

    # ==============
    # Case 2：純文字聊天（用到過去檔案摘要當上下文）
    # ==============
    if not message:
        return ChatResponse(reply="請輸入問題，或上傳圖片 / PDF 喔！")

    # 把這次的 user 訊息存起來
    chat_memory.append({"role": "user", "content": message})
    if len(chat_memory) > 50:
        chat_memory.pop(0)

    # 組 prompt（把最近的對話 + 檔案摘要都帶進去）
    system_prompt = """
你是一個自然、口語化、但回答有重點的 AI 助理，請使用繁體中文。

你可以利用：
- 對話內容
- 先前上傳檔案的「摘要」

來回答問題。

如果需要用到以前的檔案，就參考我提供的摘要，
不要再跟使用者說「看不到圖片」或「沒有 PDF」，
只要根據摘要誠實回答即可。
""".strip()

    convo: List[str] = [system_prompt]

    # 最多帶入最近 15 筆記錄（包含 user / assistant / 檔案）
    for m in chat_memory[-15:]:
        role = m.get("role")
        if role == "user":
            convo.append(f"使用者：{m['content']}")
        elif role == "assistant":
            convo.append(f"助理：{m['content']}")
        elif role == "file":
            kind = m.get("kind", "file")
            name = m.get("name", "")
            summary = m.get("summary", "")
            convo.append(
                f"（先前上傳的檔案摘要，類型：{kind}，檔名：{name}）：\n{summary}"
            )

    convo.append(f"使用者：{message}")

    try:
        ans = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=convo,
        )
        reply = ans.text.strip()
    except Exception as e:
        reply = f"❌ Chat 回答錯誤：{e}"

    chat_memory.append({"role": "assistant", "content": reply})
    if len(chat_memory) > 50:
        chat_memory.pop(0)

    return ChatResponse(reply=reply)
