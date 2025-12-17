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
"""
chat_memory：
- user / assistant：一般對話
- file：使用者上傳的圖片 / PDF，其摘要會被寫進 summary
"""
chat_memory: List[Dict[str, Union[str, Dict]]] = []

# =========================
# Prompts
# =========================
IMAGE_CLASSIFY_PROMPT = """
你是一個圖片 / 檔案類型分類助手。
請判斷這個檔案主要屬於以下哪一種：
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
請將圖片或 PDF 的主要內容摘要成 200 字以內。
請包含：
- 主要物件或文字
- 若是文件，摘要段落內容
- 若是表格，摘要欄位與重點資料
- 若是行事曆，摘要有哪些重要事件或節日（不要逐格念）
- 若是照片，描述主要場景

請用繁體中文簡潔撰寫。
"""

def build_image_answer_prompt(img_type: str) -> str:
    """
    根據檔案類型產生回答規則 Prompt
    """
    base = """
你是一個「檔案 / 圖片理解助理」。請根據使用者的問題，
只回答與問題直接相關的資訊，並使用繁體中文。

你可以：
- 參考使用者上傳檔案的摘要內容
- 視需要自行推理與使用模型內建的搜尋能力
    """

    if img_type == "calendar":
        base += """
⚠️ 規則：
1. 只回答與日期、節日、行程相關的內容。
2. 不要逐格念整個行事曆。
3. 若行事曆裡沒有相關資訊，請回答「行事曆中沒有找到這個資訊」。
"""
    elif img_type == "table":
        base += """
⚠️ 規則：
1. 依照表格摘要回答重點欄位與資料。
2. 不要逐列抄整張表。
"""
    elif img_type == "document":
        base += """
⚠️ 規則：
1. 依據文件摘要中的內容作答。
2. 優先抓出條列重點，避免過多贅詞。
"""
    else:
        base += """
⚠️ 規則：
1. 只根據檔案摘要中有提到的資訊回答。
2. 若摘要中沒有提到，請說「我在檔案摘要中沒有看到相關資訊」。
"""

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

        # 這個其實目前沒用到，只是保留以後若要存原始檔可以用
        file_b64 = base64.b64encode(file_bytes).decode()

        # Step 1：分類
        try:
            classify = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    IMAGE_CLASSIFY_PROMPT,
                    types.Part.from_bytes(file_bytes, mime),
                ],
            )
            img_type = classify.text.strip().lower()
        except Exception as e:
            img_type = "other"

        # Step 2：摘要
        try:
            summary = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    IMAGE_SUMMARY_PROMPT,
                    types.Part.from_bytes(file_bytes, mime),
                ],
            )
            img_summary = summary.text.strip()
        except Exception as e:
            img_summary = f"⚠️ 檔案摘要失敗：{e}"
        
        # Step 3：記錄到 memory
        chat_memory.append({
            "role": "file",
            "filename": image.filename,
            "mime": mime,
            "content": message or "[使用者上傳檔案]",
            "file_b64": file_b64,
            "summary": img_summary,
            "type": img_type,
        })

        # Step 4：如果有同時問問題 → 幫忙回答
        if message:
            prompt = build_image_answer_prompt(img_type)
            convo = [prompt]

            # 加入最近一份檔案摘要（通常就是這一份）
            for m in reversed(chat_memory):
                if m["role"] == "file":
                    convo.append(f"【檔案摘要】：{m['summary']}")
                    break

            convo.append(f"使用者問題：{message}")

            try:
                ans = client.models.generate_content(
                    model="gemini-2.5-flash",   # ⬅ 這裡改成 flash
                    contents=convo,
                )
                reply = ans.text.strip()
            except Exception as e:
                reply = f"⚠️ 檔案相關問題回答失敗：{e}"

            chat_memory.append({"role": "assistant", "content": reply})
            return ChatResponse(reply=reply)

        # 沒問問題，就單純回摘要
        return ChatResponse(
            reply=f"已收到檔案（{image.filename}），以下是摘要：\n\n{img_summary}"
        )

    # --------------------------------------
    # CASE 2：純文字聊天
    # --------------------------------------
    if not message:
        return ChatResponse(reply="請輸入訊息或上傳圖片喔！")

    chat_memory.append({"role": "user", "content": message})

    # 對話上下文（最近 10 則 + 一些檔案摘要）
    system_prompt = """
你是一個口語化但專業的 AI 助理，請使用繁體中文回答。

你可以同時：
- 使用對話記憶推理
- 參考使用者上傳的檔案摘要
- 視需要使用模型內建的搜尋能力查找最新資訊（不需要使用者特別要求）

回答時要：
- 先直接回答問題
- 再補充必要的背景說明
- 不要亂編造明顯錯誤的事實
"""

    convo: List[str] = [system_prompt]

    for m in chat_memory[-10:]:
        if m["role"] == "user":
            convo.append(f"使用者：{m['content']}")
        elif m["role"] == "assistant":
            convo.append(f"助理：{m['content']}")
        elif m["role"] == "file":
            convo.append(f"（之前的檔案摘要）：{m['summary']}")

    convo.append(f"使用者：{message}")

    try:
        # 這裡改用 gemini-2.5-flash（同樣可以用搜尋）
        ans = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=convo,
        )
        reply = ans.text.strip()
    except Exception as e:
        reply = f"⚠️ 伺服器呼叫 Gemini 失敗：{e}"

    chat_memory.append({"role": "assistant", "content": reply})
    return ChatResponse(reply=reply)
