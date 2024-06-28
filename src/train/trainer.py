from datasets import Dataset
from transformers import Trainer as HFTrainer, TrainingArguments
import torch.nn as nn


class Trainer:
    def __init__(self):
        pass

    def train(
        self,
        model: nn.Module,
        training_args: TrainingArguments,
        train_dataset: Dataset,
        eval_dataset: Dataset,
    ):
        trainer = HFTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        trainer.train()
