import logging
import os

import torch
import wandb

from .data import DataLoader, Datasets
from .model import ModelLoader, Models
from .train import Trainer
from .utils import TrainConfig

_LOGGER = logging.getLogger(__name__)


class Runner:
    config: TrainConfig

    def __init__(self, config_args: TrainConfig):
        self.config = config_args

        if config_args.is_ready_for_training:
            wandb.init(
                project="lora",
                config={
                    "learning_rate": config_args.learning_rate,
                    "epochs": config_args.epochs,
                },
            )
        else:
            os.environ["WANDB_MODE"] = "disabled"

    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda":
            torch.cuda.empty_cache()
        _LOGGER.info(f"Using device: {device}")

        _config = self.config

        model_loader = ModelLoader()

        # 1. Load a LoraModel (Use LoraConfig)
        tokenizer, base_model, lora_model = model_loader.load_lora_model(
            Models.EEVE_10_8B, training_config=_config
        )

        # 2. Load a dataset
        data_loader = DataLoader(
            eval_ratio=_config.eval_ratio, test_ratio=_config.test_ratio
        )

        dataset = data_loader.load_dataset(Datasets.LUCKY_VICKY)

        # 3. Train (Use TrainingArguments)
        trainer = Trainer(config=_config)
        trainer.train(
            model=lora_model,
            tokenizer=tokenizer,
            dataset=dataset,
        )

        wandb.finish()
