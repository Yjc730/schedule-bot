import speech_recognition as sr

def listen_command(timeout=5):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("🗣️ 請說指令（開始說話）")

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        audio = recognizer.listen(source, timeout=timeout)

    try:
        text = recognizer.recognize_google(audio, language="zh-TW")
        print(f"📝 你說的是：{text}")
        return text

    except sr.UnknownValueError:
        print("🤷 聽不到你在說什麼")
        return None

    except sr.RequestError as e:
        print(f"❌ 語音服務錯誤: {e}")
        return None
