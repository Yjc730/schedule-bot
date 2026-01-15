import os
import json
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

def parse_intent(command: str) -> dict:
    prompt = f"""
你是一個語音助理的 intent parser。
請將使用者指令轉成 JSON，只能輸出 JSON。

格式：
{{
  "intent": "send_email | create_event | open_app | unknown",
  "target": "",
  "extra": ""
}}

使用者指令：
{command}
"""

    response = model.generate_content(prompt)
    text = response.text.strip()

    try:
        return json.loads(text)
    except Exception:
        return {
            "intent": "unknown",
            "target": "",
            "extra": text
        }
