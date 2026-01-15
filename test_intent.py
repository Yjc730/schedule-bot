from backend.intent_parser import parse_intent

command = "幫我寄信給主管說我明天請假"
result = parse_intent(command)
print(result)
