# voice/speech_to_text.py
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

print("✅ speech_to_text.py loaded")

# 初始化模型（第一次會慢一點）
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

SAMPLE_RATE = 16000
RECORD_SECONDS = 5


def listen_and_transcribe():
    print("🎤 請開始說話（5 秒）...")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()

    audio = np.squeeze(audio)

    print("🧠 辨識中...")

    segments, _ = model.transcribe(
        audio,
        language="zh",
        beam_size=5
    )

    text = "".join([seg.text for seg in segments]).strip()
    print(f"📝 你說的是：{text}")

    return text
