# backend/intent_parser.py
import os
import json
from google import genai

# ======================
# Config
# ======================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_TEXT = os.getenv("MODEL_TEXT", "gemini-2.5-flash").strip()

DEV_MODE = os.getenv("DEV_MODE", "0") == "1"

if not DEV_MODE and not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not set")

client = genai.Client(api_key=GEMINI_API_KEY) if not DEV_MODE else None

# ======================
# Unified Prompt
# ======================
SYSTEM_PROMPT = """
你是一個「語音助理的語意修復 + intent parser」。

使用者的語音轉文字可能有：
- 錯字
- 誤聽
- 同音錯誤（例如：寄信 → 記性）
- 詞語顛倒

請你完成以下工作：
1. 先在心中修正語意（不要輸出修正句）
2. 根據修正後的語意，判斷 intent
3. 輸出 intent 與 slots

⚠️ 嚴格規則：
- 只輸出 JSON
- 不要任何解釋
- 不要 markdown
- 不要多餘文字

JSON 格式：
{
  "intent": "<intent_name>",
  "slots": { ... }
}

可用 intent：
- send_email
- open_app
- unknown

send_email slots 範例：
{
  "recipient": "主管",
  "body": "我明天請假"
}
"""

# ======================
# Main function
# ======================
def parse_intent(command: str) -> dict:
    # ===== DEV MODE：不打 API（本地測試用）=====
    if DEV_MODE:
        if any(word in command for word in ["寄信", "寄", "信"]):
            return {
                "intent": "send_email",
                "slots": {
                    "recipient": "主管" if "主管" in command else None,
                    "body": command.replace("幫我", "").replace("寄信給主管", "").strip()
                }
            }
        return {"intent": "unknown", "slots": {}}

    # ===== PROD MODE：Gemini 語意修復 + intent =====
    prompt = f"""
{SYSTEM_PROMPT}

使用者說：
「{command}」
"""

    response = client.models.generate_content(
        model=MODEL_TEXT,
        contents=prompt
    )

    text = response.text.strip()

    # 防禦：清掉 ```json
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except Exception as e:
        return {
            "intent": "unknown",
            "raw": response.text,
            "error": str(e)
        }
