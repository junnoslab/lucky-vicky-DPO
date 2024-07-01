import logging

from transformers import pipeline
import torch

from .data import DataLoader, Datasets
from .model import ModelLoader, Models
from .train import Trainer
from .utils import TrainConfig

_LOGGER = logging.getLogger(__name__)


class Runner:
    config: TrainConfig

    def __init__(self, config_args: TrainConfig):
        self.config = config_args

    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _LOGGER.info(f"Using device: {device}")

        model_loader = ModelLoader()

        # 1. Load a LoraModel (Use LoraConfig)
        tokenizer, base_model, lora_model = model_loader.load_lora_model(
            Models.BLOSSOM, training_config=self.config
        )

        # 2. Load a dataset
        data_loader = DataLoader()

        dataset = data_loader.load_dataset(Datasets.LUCKY_VICKY)

        # 3. Train (Use TrainingArguments)
        trainer = Trainer(config=self.config, device=device)
        pretrained_model = trainer.train(
            model=lora_model,
            tokenizer=tokenizer,
            dataset=dataset,
        )

        print(pretrained_model)

        _pipeline = pipeline(
            "text-generation", model=pretrained_model, tokenizer=tokenizer
        )
        _pipeline.model.eval()

        messages = [
            {"role": "user", "content": "월요일부터 코감기에 걸려서 머리가 아파."}
        ]

        prompt = _pipeline.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        terminators = [
            _pipeline.tokenizer.eos_token_id,
            _pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        outputs = _pipeline(
            prompt,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        print(outputs[0]["generated_text"])
