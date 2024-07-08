DEVICE_MAP: str = "balanced"

INSTRUCTION_TEMPLATE: str = "\n### Instruction:"
QUESTION_TEMPLATE: str = "\n### Question:"
ANSWER_TEMPLATE: str = "\n### Answer:"

PROMPT_TEMPLATE: str = """
### Instruction:
다음과 같은 형식으로 불안하거나 불행한 일을 표현하는 문장을 긍정적인 표현으로 변환해주세요.
- 첫 문장에는 상황과 감정을 표현한다.
- 두 번째 문장에는 현재 상황보다 더 좋거나 덜 좋은 상황을 설명한다.
- 세번째 문장에는 더 좋거나 덜 좋은 상황 모두 별로니까 결론적으로 현재 상황이 가장 긍정적이라는 것을 설명한다.
- 마지막 문장은 고정적으로 '완전 럭키비키잔앙🍀'을 사용해 마친다.
- 감탄사와 이모지를 적극적으로 사용한다.

### Question:
{QUESTION}

### Answer:
{ANSWER}
"""

# {EXAMPLE_TEMPLATE}
# {QUESTION_TEMPLATE}
# 내 앞에서 50% 세일하는 옷이 품절됐어
# {ANSWER_TEMPLATE}
# 내 앞에서 50% 세일하는 옷이 품절된 거 있지!! 정말 아쉬워!!
# 근데 만약 그 옷을 샀다면, 집에 비슷한 옷이 있다는 걸 나중에 알게 돼서 후회했을 거야. 돈 아끼고 다른 예쁜 옷 살 기회가 생겼잖아!
# 그래서 딱 지금 품절된 게 최고야 🤭🤭 완전 럭키비키잔앙🍀
