from peft import PeftConfig, PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
import torch

from train.model import Models
from train.utils import SYSTEM_PROMPT

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = Models.BLOSSOM.value

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    base_model.eval()
    base_model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    adapter_path = "res/lora"

    peft_config = PeftConfig.from_pretrained(adapter_path)

    adapted_model = PeftModel.from_pretrained(
        model=base_model, model_id=adapter_path, adapter_name="lora", config=peft_config
    )
    adapted_model.set_adapter("lora")
    adapted_model.to(device)
    adapted_model.eval()
    print(adapted_model.active_adapter)

    adapted_pipeline = pipeline(
        "text-generation",
        model=adapted_model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    base_pipeline = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    terminators = [
        adapted_pipeline.tokenizer.eos_token_id,
        adapted_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    while True:
        text = input("☘️ ")

        prompt = adapted_pipeline.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

        adapted_outputs = adapted_pipeline(
            prompt,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        base_outputs = base_pipeline(
            prompt,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        print(adapted_outputs[0]["generated_text"][len(prompt) :])
        print(base_outputs[0]["generated_text"][len(prompt) :])
