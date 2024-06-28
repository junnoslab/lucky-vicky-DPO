from datasets import Dataset
from transformers import Trainer as HFTrainer, TrainingArguments
import torch.nn as nn

from ..utils import TrainConfig


class Trainer:
    training_args: TrainingArguments

    def __init__(self, config: TrainConfig):
        self.training_args = TrainingArguments(task_type="CAUSAL_LM")

    def train(
        self,
        model: nn.Module,
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
