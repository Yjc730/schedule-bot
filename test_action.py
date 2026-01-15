from backend.action_router import route_action

# 模擬 intent parser 的輸出
intent_data = {
    "intent": "send_email",
    "slots": {
        "recipient": "主管",
        "body": "我明天請假"
    }
}

result = route_action(intent_data)
print("✅ Result:", result)
