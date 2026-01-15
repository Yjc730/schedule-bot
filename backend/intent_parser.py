# backend/intent_parser.py
import os
import json

from google import genai

# ======================
# Config
# ======================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
MODEL_TEXT = os.getenv("MODEL_TEXT", "gemini-2.5-flash").strip()

if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not set")

client = genai.Client(api_key=GEMINI_API_KEY)

# ======================
# System Prompt
# ======================
SYSTEM_PROMPT = """
你是一個語音助理的 intent parser。
請「只輸出 JSON」，不要任何解釋文字。

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

# ======================
# Main function
# ======================
def parse_intent(command: str) -> dict:
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

    # 🔧 去掉 ```json ``` 包裝（非常重要）
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
