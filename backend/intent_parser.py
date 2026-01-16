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
# Prompt 1：語意修復
# ======================
FIX_PROMPT = """
你是一個語音指令修正器。

使用者的語音轉文字可能有：
- 錯字
- 誤聽
- 詞語顛倒
- 同音錯誤（例如：寄信 → 記性）

請根據語意，修正成一個「合理、自然的人類指令句」。

⚠️ 規則：
- 只輸出修正後的句子
- 不要解釋
- 不要加引號
"""

# ======================
# Prompt 2：Intent Parser
# ======================
INTENT_PROMPT = """
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
    # -------- Step 1：語意修復 --------
    fix_response = client.models.generate_content(
        model=MODEL_TEXT,
        contents=[
            FIX_PROMPT,
            f"原始語音轉文字：{command}"
        ]
    )

    fixed_command = fix_response.text.strip()
    print(f"🛠 修正後指令：{fixed_command}")

    # -------- Step 2：Intent 判斷 --------
    intent_prompt = f"""
{INTENT_PROMPT}

使用者說：
「{fixed_command}」
"""

    response = client.models.generate_content(
        model=MODEL_TEXT,
        contents=intent_prompt
    )

    text = response.text.strip()

    # 🔧 去掉 ```json ``` 包裝
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
            "fixed_command": fixed_command,
            "error": str(e)
        }
