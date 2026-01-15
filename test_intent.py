from intent_parser import parse_intent

while True:
    cmd = input("請輸入指令（exit 離開）：")
    if cmd == "exit":
        break

    result = parse_intent(cmd)
    print("🧠 Intent Result:")
    print(result)
