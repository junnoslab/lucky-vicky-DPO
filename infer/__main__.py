import logging
import time

from peft import PeftConfig, PeftModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
    TextStreamer,
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

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        device_map=_DEVICE_MAP,
    )
    base_model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=_DEVICE_MAP)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    adapter_path = "res/"

    peft_config = PeftConfig.from_pretrained(adapter_path)
    peft_config.inference_mode = True

    adapted_model = PeftModel.from_pretrained(
        model=base_model,
        model_id=adapter_path,
        adapter_name="TrainAdapter",
        config=peft_config,
        is_trainable=False,
        device_map=_DEVICE_MAP,
    )
    del base_model
    adapted_model.set_adapter("TrainAdapter")
    _LOGGER.info(f"Using adapter: {adapted_model.active_adapter}")

    adapted_model.generation_config.cache_implementation = "static"

    compiled_model = torch.compile(
        adapted_model, mode="reduce-overhead", fullgraph=True
    )
    compiled_model.eval()

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
    ]

    config = GenerationConfig(
        top_p=0.9,
        temperature=0.6,
        num_return_sequences=1,
        do_sample=True,
    )

    while True:
        text = input("> Lucky Vicky 😊☘️ ")

        start = time.time()

        instruction = PROMPT_TEMPLATE.format(QUESTION=text, ANSWER="")
        input_ids = tokenizer(instruction, return_tensors="pt").input_ids.to(device)

        streamer = TextStreamer(tokenizer)

        _ = compiled_model.generate(
            input_ids,
            generation_config=config,
            streamer=streamer,
            max_new_tokens=256,
            eos_token_id=terminators,
        )

        end = time.time()
        _LOGGER.info(f"Inference time: {end - start:.2f} sec.")
