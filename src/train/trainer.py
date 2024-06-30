from datasets import Dataset
from transformers import Trainer as HFTrainer, TrainingArguments, AutoTokenizer
import torch.nn as nn

from ..utils import TrainConfig


class Trainer:
    training_args: TrainingArguments

    def __init__(self, config: TrainConfig):
        self.training_args = TrainingArguments(
            output_dir=config.output_dir,
            eval_strategy="epoch",
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            learning_rate=config.learning_rate,
            lr_scheduler_type=config.lr_scheduler_type,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            save_steps=config.save_steps,
            logging_steps=10,
        )

    def train(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        train_dataset: Dataset,
        eval_dataset: Dataset,
    ):
        trainer = HFTrainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trainer.train()

        trainer.save_model(self.training_args.output_dir)
