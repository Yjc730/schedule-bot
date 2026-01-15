from backend.intent_parser import parse_intent
from backend.action_router import route_action

command = "我明天請假，幫我寄信給主管"  # 目前來自語音

intent_data = parse_intent(command)
route_action(intent_data)
