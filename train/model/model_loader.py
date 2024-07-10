from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
import torch

from .models import Models
from ..utils import TrainConfig, DEVICE_MAP


class ModelLoader:
    """
    A class responsible for loading models and returning tokenizer and model instances.

    Attributes:
        None

    Methods:
        load_tokenizer_and_model: Load the specified model and return the tokenizer and model instances.
    """

    def __init__(self) -> None:
        pass

    def load_tokenizer_and_model(
        self, model: Models
    ) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        """
        Load the specified model and return the tokenizer and model instances.

        Args:
            model (Models): The enum value representing the model to be loaded.

        Returns:
            tuple[AutoTokenizer, AutoModelForCausalLM]: A tuple containing the tokenizer and model instances.
        """
        _tokenizer = AutoTokenizer.from_pretrained(model.value, device_map=DEVICE_MAP)
        _tokenizer.padding_side = "right"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
        _model = AutoModelForCausalLM.from_pretrained(
            model.value,
            torch_dtype=model.dtype,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            device_map=DEVICE_MAP,
        )
        _model.config.use_cache = False
        return _tokenizer, _model

    def load_lora_model(
        self, model: Models, training_config: TrainConfig
    ) -> tuple[AutoTokenizer, AutoModelForCausalLM, LoraConfig]:
        tokenizer, base_model = self.load_tokenizer_and_model(model)
        config = LoraConfig(
            task_type="CAUSAL_LM",
            r=training_config.rank,
            lora_alpha=training_config.lora_alpha,
            lora_dropout=training_config.lora_dropout,
            bias=training_config.bias,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        return tokenizer, base_model, config
