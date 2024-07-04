from datasets import Dataset
from transformers import (
    Trainer as HFTrainer,
    TrainingArguments,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
import torch
import torch.nn as nn

from ..utils import TrainConfig, SYSTEM_PROMPT


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
            dataloader_num_workers=config.dataloader_num_workers,
            save_steps=config.save_steps,
            logging_steps=10,
        )

    def train(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset,
    ) -> PreTrainedModel:
        # Setup dataset
        def preprocess(dataset: Dataset):
            dataset["input"] = (
                f"{SYSTEM_PROMPT} \n###Input: {dataset['input']} \n###Output:{dataset['output']}"
            )
            return tokenizer(
                dataset["input"],
                text_target=dataset["output"],
                padding="max_length",
                max_length=512,
                truncation=True,
            )

        tokenized_datasets = dataset.map(preprocess, batch_size=True)

        # https://github.com/KoJLabs/StrategicDataOrdering/blob/dbdabc2ee523e5f42b7b2cffa74d731a8df7281f/train.py#L117C5-L117C13
        # DataCollator 참고

        trainer = HFTrainer(
            model=model,
            args=self.training_args,
            train_dataset=tokenized_datasets["train"],
        )
        trainer.train()

        trainer.model.save_pretrained(self.training_args.output_dir)
        trainer.save_model(self.training_args.output_dir)
        trainer.model.config.save_pretrained(self.training_args.output_dir)

        return trainer.model
