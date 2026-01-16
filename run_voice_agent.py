# run_voice_agent.py

from voice.speech_to_text import listen_and_transcribe

def run_voice_agent():
    print("🤖 Voice Agent 啟動中...\n")

    # 1. 聽使用者說話
    text = listen_and_transcribe()

    if not text:
        print("⚠️ 沒有聽到任何內容")
        return

    # 2. 假助理回應（目前先不接 LLM）
    reply = f"你剛剛說的是：{text}"

    # 3. 輸出回應
    print("\n🤖 助理回應：")
    print(reply)


if __name__ == "__main__":
    run_voice_agent()
