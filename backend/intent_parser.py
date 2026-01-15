# backend/intent_parser.py
import os
import json
import google.genai as genai

# =====================
# Config
# =====================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_INTENT = os.getenv("MODEL_INTENT", "gemini-2.5-flash").strip()

if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY 未設定")

client = genai.Client(api_key=GEMINI_API_KEY)

# =====================
# Intent Parser
# =====================
def parse_intent(command: str) -> dict:
    """
    將自然語言指令轉為 intent + slots
    """

    system_prompt = """
你是一個「指令解析器」，只輸出 JSON，不要解釋。

請將使用者指令轉成以下格式：
{
  "intent": "<動作>",
  "slots": { ... }
}

可用 intent：
- send_email
- create_calendar
- open_app
- unknown

規則：
- 若是寄信，intent=send_email，slots 包含 to, content
- 若是行事曆，intent=create_calendar，slots 包含 date, time, title
- 若只是開 App，intent=open_app，slots 包含 app
- 無法判斷就 intent=unknown
"""

    prompt = f"""
{system_prompt}

使用者指令：
「{command}」
"""

    response = client.models.generate_content(
        model=MODEL_INTENT,
        contents=prompt,
    )

    text = response.text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "intent": "unknown",
            "slots": {},
            "raw": text
        }
