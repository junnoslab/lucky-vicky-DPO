from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraModel, LoraConfig

from .models import Models


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
            model.value, torch_dtype=model.dtype
        )
        return _tokenizer, _model

    def load_lora_model(
        self, model: Models, config: LoraConfig, adapter_name: str = "lora"
    ) -> LoraModel:
        """
        Load the specified model and return the LoraModel instance.

        Args:
            model (Models): The enum value representing the model to be loaded.

        Returns:
            LoraModel: The LoraModel instance.
        """
        _, _model = self.load_tokenizer_and_model(model)
        return LoraModel(_model, config=config, adapter_name=adapter_name)
