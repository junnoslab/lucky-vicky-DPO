from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraModel, LoraConfig

from .models import Models
from ..utils import TrainConfig


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
        _tokenizer = AutoTokenizer.from_pretrained(model.value)
        _model = AutoModelForCausalLM.from_pretrained(
            model.value, torch_dtype=model.dtype, device_map="auto"
        )
        return _tokenizer, _model

    def load_lora_model(
        self, model: Models, training_config: TrainConfig, adapter_name: str = "lora"
    ) -> LoraModel:
        """
        Load the specified model and return the LoraModel instance.

        Args:
            model (Models): The enum value representing the model to be loaded.

        Returns:
            LoraModel: The LoraModel instance.
        """
        _, _model = self.load_tokenizer_and_model(model)
        config = LoraConfig(
            task_type="CAUSAL_LM",
            r=training_config.rank,
            lora_alpha=training_config.lora_alpha,
            lora_dropout=training_config.lora_dropout,
            bias=training_config.bias,
            init_lora_weights="gaussian",
        )
        lora_model = get_peft_model(
            model=_model, peft_config=config, adapter_name=adapter_name
        )
        return lora_model
