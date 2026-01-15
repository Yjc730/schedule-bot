# test_intent.py
from backend.intent_parser import parse_intent

while True:
    cmd = input("🗣 請輸入指令：")
    if cmd in ("exit", "quit"):
        break

    result = parse_intent(cmd)
    print("🤖 Intent result:")
    print(result)
