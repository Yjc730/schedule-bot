import os
import json
import google.generativeai as genai

# 初始化 Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")


def parse_intent(command: str) -> dict:
    """
    把使用者說的話轉成結構化 intent
    """
    prompt = f"""
你是一個語音助理的意圖解析器。
請把使用者指令轉成 JSON，不要解釋。

指令：
「{command}」

請只回傳 JSON，格式如下：
{{
  "intent": "...",
  "target": "...",
  "extra": "..."
}}
"""

    response = model.generate_content(prompt)
    text = response.text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "intent": "unknown",
            "raw": text
        }
