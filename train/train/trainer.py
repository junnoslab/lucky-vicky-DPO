from datasets import Dataset
from transformers import (
    Trainer as HFTrainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
import torch
import torch.nn as nn

from ..data import DataCollator
from ..utils import TrainConfig


class Trainer:
    device: torch.device
    training_args: TrainingArguments

    def __init__(self, config: TrainConfig, device: torch.device):
        self.device = device
        self.training_args = TrainingArguments(
            output_dir=config.output_dir,
            evaluation_strategy="epoch",
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            learning_rate=config.learning_rate,
            lr_scheduler_type=config.lr_scheduler_type,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            dataloader_pin_memory=False,
            save_steps=config.save_steps,
            logging_steps=10,
        )

    def train(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset,
    ) -> PreTrainedModel:
        # Setup model
        model = model.to(self.device)

        # Setup dataset
        def tokenize_function(examples: Dataset):
            return tokenizer(
                examples["input"],
                text_target=examples["output"],
                padding="max_length",
                max_length=512,
                truncation=True,
            )

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        data_collator = DataCollator(tokenizer=tokenizer, device=self.device)

        trainer = HFTrainer(
            model=model,
            args=self.training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["eval"],
            data_collator=data_collator,
        )
        trainer.train()

        trainer.model.save_pretrained(self.training_args.output_dir)
        trainer.save_model(self.training_args.output_dir)
        trainer.model.config.save_pretrained(self.training_args.output_dir)

        return trainer.model
