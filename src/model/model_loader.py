from transformers import AutoTokenizer, AutoModelForCausalLM

from .models import Models


class ModelLoader:
    def __init__(self) -> None:
        pass

    def load_model(self, model: Models) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        _tokenizer = AutoTokenizer.from_pretrained(model.value)
        _model = AutoModelForCausalLM.from_pretrained(model.value)
        return _tokenizer, _model
