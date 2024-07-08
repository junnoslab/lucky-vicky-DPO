from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("yanolja/EEVE-Korean-10.8B-v1.0")
tokenizer = AutoTokenizer.from_pretrained("yanolja/EEVE-Korean-10.8B-v1.0")

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

PROMPT = PROMPT_TEMPLATE.format(
    QUESTION="내 앞에서 50% 세일하는 옷이 품절됐어", ANSWER=""
)


def print_tokens_with_ids(txt):
    tokens = tokenizer.tokenize(txt, add_special_tokens=False)
    token_ids = tokenizer.encode(txt, add_special_tokens=False)
    print(list(zip(tokens, token_ids)))


print_tokens_with_ids(PROMPT)
print_tokens_with_ids(INSTRUCTION_TEMPLATE)
print_tokens_with_ids(ANSWER_TEMPLATE)

instruction_template_ids = tokenizer.encode(
    INSTRUCTION_TEMPLATE, add_special_tokens=False
)[2:]
answer_template_ids = tokenizer.encode(ANSWER_TEMPLATE, add_special_tokens=False)[2:]
