from backend.intent_parser import parse_intent

if __name__ == "__main__":
    command = "幫我寄信給主管"
    result = parse_intent(command)
    print("🎯 Intent Result:")
    print(result)
