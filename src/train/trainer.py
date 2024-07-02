from datasets import Dataset
from transformers import (
    Trainer as HFTrainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
import evaluate
import numpy as np
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
            inputs = tokenizer(
                examples["input"],
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            targets = tokenizer(
                examples["output"],
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            inputs["labels"] = targets["input_ids"]
            return inputs

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

        data_collator = DataCollator(tokenizer=tokenizer, device=self.device)

        def compute_metrics(eval_pred):
            metric = evaluate.load("accuracy")

            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        trainer = HFTrainer(
            model=model,
            args=self.training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["eval"],
            data_collator=data_collator,
            # compute_metrics=compute_metrics,
        )
        trainer.train()

        trainer.model.save_pretrained(self.training_args.output_dir)
        trainer.save_model(self.training_args.output_dir)
        trainer.model.config.save_pretrained(self.training_args.output_dir)

        return trainer.model
