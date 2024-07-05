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
from train.utils import SYSTEM_PROMPT

_LOGGER = logging.getLogger(__name__)

_DEVICE_MAP = "balanced"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = Models.EEVE_10_8B.value

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        device_map=_DEVICE_MAP,
    )
    base_model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, device_map=_DEVICE_MAP
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(["###Input:", "###Output:", "###Instruction:", "###Example:"])
    base_model.resize_token_embeddings(len(tokenizer))

    adapter_path = "res/lora"

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
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=_DEVICE_MAP,
    )

    terminators = [
        adapted_pipeline.tokenizer.eos_token_id,
        adapted_pipeline.tokenizer.convert_tokens_to_ids("<|im_end|>"),
    ]

    while True:
        text = input("☘️ ")

        start = time.time()

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
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        end = time.time()

        _LOGGER.info(adapted_outputs[0]["generated_text"][len(prompt) :])
        _LOGGER.info(f"Inference time: {end - start:.5f} sec.")
