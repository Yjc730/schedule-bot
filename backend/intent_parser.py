# backend/intent_parser.py
import os
import json
from google import genai

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_TEXT = os.getenv("MODEL_TEXT", "gemini-2.5-flash").strip()

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """
你是一個語音助理的 intent parser。
請只輸出 JSON，不要任何解釋文字。

格式：
{
  "intent": "<intent_name>",
  "slots": { ... }
}

可用 intent：
- send_email
- open_app
- unknown
"""

def parse_intent(command: str) -> dict:
    prompt = f"""
{SYSTEM_PROMPT}

使用者說：
「{command}」
"""

    resp = client.models.generate_content(
        model=MODEL_TEXT,
        contents=prompt
    )

    text = resp.text.strip()

    try:
        return json.loads(text)
    except Exception:
        return {
            "intent": "unknown",
            "raw": text
        }
