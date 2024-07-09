import logging
import time

from peft import PeftConfig, PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
import torch

from train.model import Models
from train.utils.templates import PROMPT_TEMPLATE

_LOGGER = logging.getLogger(__name__)

_DEVICE_MAP = "balanced"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Models.EEVE_10_8B
    model_name = model.value

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        device_map=_DEVICE_MAP,
    )
    base_model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=_DEVICE_MAP)

    adapter_path = f"res/{model_name.split('/')[-1]}"

    peft_config = PeftConfig.from_pretrained(adapter_path)

    adapted_model = PeftModel.from_pretrained(
        model=base_model,
        model_id=adapter_path,
        adapter_name="lora",
        config=peft_config,
        is_trainable=False,
        device_map=_DEVICE_MAP,
    )
    del base_model
    adapted_model.set_adapter("lora")
    _LOGGER.info(f"Using adapter: {adapted_model.active_adapter}")

    adapted_model.generation_config.cache_implementation = "static"

    compiled_model = torch.compile(
        adapted_model, mode="reduce-overhead", fullgraph=True
    )
    compiled_model.eval()

    adapted_pipeline = pipeline(
        "text-generation",
        model=compiled_model,
        tokenizer=tokenizer,
        device_map=_DEVICE_MAP,
    )

    terminators = [
        adapted_pipeline.tokenizer.eos_token_id,
        adapted_pipeline.tokenizer.convert_tokens_to_ids("<|im_end|>"),
    ]

    while True:
        text = input("☘️ ")

        start = time.time()

        instruction = PROMPT_TEMPLATE.format(QUESTION=text, ANSWER="")
        print(instruction)

        # prompt = adapted_pipeline.tokenizer.apply_chat_template(
        #     [
        #         {"role": "system", "content": instruction},
        #         {"role": "user", "content": text},
        #     ],
        #     tokenize=False,
        #     add_generation_prompt=True,
        # )

        adapted_outputs = adapted_pipeline(
            instruction,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        end = time.time()

        _LOGGER.info(adapted_outputs[0]["generated_text"])
        _LOGGER.info(f"Inference time: {end - start:.2f} sec.")
