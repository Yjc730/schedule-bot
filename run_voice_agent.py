# run_voice_agent.py

from actions.send_email import send_email_via_outlook
from backend.intent_parser import parse_intent

# ===== 聯絡人 =====
CONTACTS = {
    "主管": "boss@example.com",
    "老闆": "boss@example.com",
}

# ===== 確認 / 取消 關鍵字（你原本缺的）=====
CONFIRM_WORDS = ["對", "是", "沒錯", "確認", "好", "可以"]
CANCEL_WORDS = ["不要", "取消", "不是", "算了"]

# ===== 狀態 =====
pending_action = None


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
            recipient=recipient_email,
            subject="通知",
            body=body
        )

    else:
        print("🤷 不知道怎麼處理這個 intent")


def run_voice_agent():
    global pending_action

    print("🚀 Voice Agent started")

    while True:
        print("👂 Wake word detected!")

        # 👉 目前用 input() 測試（非常正確）
        command = input("⌨️ 輸入指令：").strip()
        if not command:
            print("⚠️ 沒輸入內容，回到待命")
            continue

        print("📝 Command:", command)

        # ===== 狀態 1：沒有待確認動作 =====
        if pending_action is None:
            intent_data = parse_intent(command)
            intent = intent_data.get("intent")

            if intent == "send_email":
                pending_action = intent_data

                recipient = intent_data["slots"].get("recipient", "對方")
                body = intent_data["slots"].get("body", "")

                print(
                    f"🗣️ 你是要寄信給「{recipient}」，"
                    f"內容是「{body}」，對嗎？"
                )
                continue

            else:
                route_action(intent_data)
                print("🔁 回到待命狀態\n")
                continue

        # ===== 狀態 2：等待確認 / 取消 =====
        else:
            if any(word in command for word in CONFIRM_WORDS):
                print("✅ 使用者確認，執行動作")
                route_action(pending_action)
                pending_action = None

            elif any(word in command for word in CANCEL_WORDS):
                print("❌ 使用者取消操作")
                pending_action = None

            else:
                print("🤔 我沒聽懂，請回答「對」或「取消」")
                continue

            print("🔁 回到待命狀態\n")


if __name__ == "__main__":
    run_voice_agent()
