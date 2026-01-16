pending_action = None

def run_voice_agent():
    global pending_action

    print("🚀 Voice Agent started")

    while True:
        print("👂 Wake word detected!")

        command = listen_and_transcribe()
        if not command:
            print("⚠️ 沒聽清楚，回到待命")
            continue

        print("📝 Command:", command)

        # ===== 狀態 1：目前沒有待確認的動作 =====
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

        # ===== 狀態 2：正在等使用者確認 =====
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

            print("🔁 回到待命狀態\n")
