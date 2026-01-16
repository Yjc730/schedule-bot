# run_voice_agent.py

from actions.send_email import send_email_via_outlook
from backend.intent_parser import parse_intent
from voice.speech_to_text import listen_and_transcribe
#from voice.wakeword import listen_wake_word

CONTACTS = {
    "主管": "boss@example.com",
    "老闆": "boss@example.com",
}

def route_action(intent_data: dict):
    intent = intent_data.get("intent")
    slots = intent_data.get("slots", {})

    print("🧭 Routing intent:", intent)
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
            to=recipient_email,
            subject="通知",
            body=body
        )

    else:
        print("🤷 不知道怎麼處理這個 intent")

def run_voice_agent():
    print("🚀 Voice Agent started")

    while True:
        # A-3-1：等待喚醒詞
        #listen_wake_word()

        print("👂 Wake word detected!")

        # A-2：語音 → 文字
        command = listen_and_transcribe()
        if not command:
            print("⚠️ 沒聽清楚，回到待命")
            continue

        print("📝 Command:", command)

        # B：Intent
        intent_data = parse_intent(command)

        # C：Action
        route_action(intent_data)

        print("🔁 回到待命狀態\n")

if __name__ == "__main__":
    run_voice_agent()
