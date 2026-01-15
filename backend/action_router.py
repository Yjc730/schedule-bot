# backend/action_router.py
import os

def route_action(intent_data: dict):
    """
    根據 intent + slots，執行對應行為
    """
    intent = intent_data.get("intent", "unknown")
    slots = intent_data.get("slots", {})

    print(f"🧭 Routing intent: {intent}")
    print(f"📦 Slots: {slots}")

    if intent == "send_email":
        return handle_send_email(slots)

    elif intent == "open_app":
        return handle_open_app(slots)

    else:
        return handle_unknown(slots)


def handle_send_email(slots: dict):
    recipient = slots.get("recipient", "未知對象")
    body = slots.get("body", "")

    # ⚠️ 目前是 mock（假寄信）
    print("📧 [MOCK] 寄送 Email")
    print(f"➡️ 收件者：{recipient}")
    print(f"📝 內容：{body}")

    # 之後可以接 Gmail API / Outlook
    return {
        "status": "ok",
        "action": "send_email",
        "recipient": recipient
    }


def handle_open_app(slots: dict):
    app_name = slots.get("app", "Google Chrome")

    print(f"🚀 開啟應用程式：{app_name}")

    # macOS
    os.system(f'open -a "{app_name}"')

    return {
        "status": "ok",
        "action": "open_app",
        "app": app_name
    }


def handle_unknown(slots: dict):
    print("🤷 我不確定你要做什麼")

    return {
        "status": "unknown",
        "action": "none"
    }
