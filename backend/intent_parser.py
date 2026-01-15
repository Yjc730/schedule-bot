import os
import json
import google.genai as genai

# 初始化 Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SYSTEM_PROMPT = """
你是一個語音助理的意圖解析器。
請將使用者輸入的中文指令，轉換為 JSON。

可用 intent 僅限以下：
- send_email
- create_calendar_event
- open_app
- unknown

JSON 格式必須如下：
{
  "intent": "...",
  "target": "...",
  "content": "..."
}

規則：
- 如果沒有對象，target 可為 null
- 如果沒有內容，content 可為 null
- 只輸出 JSON，不要任何多餘文字
"""

def parse_intent(command: str) -> dict:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {"role": "system", "parts": [SYSTEM_PROMPT]},
            {"role": "user", "parts": [command]}
        ]
    )

    text = response.text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "intent": "unknown",
            "target": None,
            "content": None
        }
