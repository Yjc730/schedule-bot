import os
import time
import pvporcupine
import pyaudio
import struct

print("✅ wakeword.py loaded")

def listen_wake_word():
    print("🎙 listen_wake_word() called")
    access_key = os.getenv("PICOVOICE_ACCESS_KEY")
    if not access_key:
        print("❌ PICOVOICE_ACCESS_KEY 未設定")
        return

    print("🎙 Listening for wake word: hey computer")

    porcupine = pvporcupine.create(
        access_key=access_key,
        keywords=["computer"]  # 內建關鍵字，先測 pipeline
    )

    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length,
    )

    try:
        while True:
            pcm = stream.read(
                porcupine.frame_length,
                exception_on_overflow=False
            )
            pcm = struct.unpack_from(
                "h" * porcupine.frame_length,
                pcm
            )

            if result >= 0:
    print("🔥 Wake word detected! Opening Outlook...")
    os.system('open -a "Microsoft Outlook"')
    break


    except KeyboardInterrupt:
        print("👋 停止監聽")

    finally:
        stream.close()
        pa.terminate()
        porcupine.delete()
