from fastapi import APIRouter
from backend.intent_parser import parse_intent
from actions.send_email import send_email_via_outlook

router = APIRouter()

CONTACTS = {
    "主管": "boss@example.com",
    "老闆": "boss@example.com",
}

@router.post("/voice-command")
async def voice_command(payload: dict):
    """
    Web 專用語音指令入口
    payload = { "text": "幫我寄信給主管說我明天請假" }
    """
    text = payload.get("text", "").strip()
    if not text:
        return {"reply": "我沒有聽到內容喔"}

    intent_data = parse_intent(text)
    intent = intent_data.get("intent")
    slots = intent_data.get("slots", {})

    # ===== 行為型指令 =====
    if intent == "send_email":
        recipient = slots.get("recipient")
        body = slots.get("body", "")

        if not recipient:
            return {"reply": "你要寄給誰？"}

        email = CONTACTS.get(recipient)
        if not email:
            return {"reply": f"我找不到 {recipient}"}

        send_email_via_outlook(email, body)

        return {
            "reply": f"✅ 已幫你寄信給 {recipient}"
        }

    # ===== 不是行為，就當聊天 =====
    from backend.main import chat
    fake_form = {"message": text}
    result = await chat(**fake_form)

    return {"reply": result.reply}
