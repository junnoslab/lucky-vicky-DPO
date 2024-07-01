import logging

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
        trainer.train(
            model=lora_model,
            tokenizer=tokenizer,
            dataset=dataset,
        )
