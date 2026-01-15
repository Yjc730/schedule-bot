# run_voice_agent.py

from actions.send_email import send_email_via_outlook
from backend.intent_parser import parse_intent

# 📇 聯絡人對照表（先寫死）
CONTACTS = {
    "主管": "boss@example.com",
    "老闆": "boss@example.com",
}

def route_action(intent_data: dict):
    """
    根據 intent 執行對應行為
    """
    intent = intent_data.get("intent")
    slots = intent_data.get("slots", {})

    print("🚦 Routing intent:", intent)
    print("📦 Slots:", slots)

    if intent == "send_email":
        recipient_name = slots.get("recipient")
        body = slots.get("body", "")

        if not recipient_name:
            print("❌ 缺少收件人")
            return

        recipient_email = CONTACTS.get(recipient_name)

        if not recipient_email:
            print(f"❌ 找不到聯絡人：{recipient_name}")
            return

        send_email_via_outlook(
            recipient_email=recipient_email,
            body=body
        )

    else:
        print("🤷 尚未支援的 intent:", intent)


# ======================
# 🔊 模擬語音輸入（現在）
# ======================
if __name__ == "__main__":
    command = "我明天請假，幫我寄信給主管"
    print("🎤 COMMAND =", command)

    intent_data = parse_intent(command)
    print("🧠 INTENT_DATA =", intent_data)

    route_action(intent_data)
