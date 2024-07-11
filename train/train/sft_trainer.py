import os

from datasets import Dataset
from peft.config import PeftConfig
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer as HFSFTTrainer
import torch.nn as nn

from ..utils import TrainConfig
from ..utils.secrets import HUGGINGFACE_HUB_TOKEN
from ..utils.templates import (
    INSTRUCTION_TEMPLATE,
    ANSWER_TEMPLATE,
    PROMPT_TEMPLATE,
)


class SFTTrainer:
    training_args: SFTConfig

    def __init__(self, config: TrainConfig):
        self.training_args = SFTConfig(
            output_dir=config.output_dir,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            max_seq_length=config.max_seq_length,
            learning_rate=config.learning_rate,
            lr_scheduler_type=config.lr_scheduler_type,
            optim=config.optimizer_type,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            warmup_ratio=config.warmup_ratio,
            dataloader_pin_memory=False,
            dataloader_num_workers=config.dataloader_num_workers,
            evaluation_strategy="no",
            save_strategy="steps",
            save_steps=config.save_steps,
            save_total_limit=2,
            logging_strategy="steps",
            logging_steps=config.logging_steps,
            push_to_hub=config.push_to_hub is not None,
            hub_model_id=config.push_to_hub,
            hub_token=HUGGINGFACE_HUB_TOKEN,
        )

    def train(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset,
        peft_config: PeftConfig,
    ) -> PreTrainedModel:
        # Setup tokenizer
        tokenizer.padding_side = "right"

        def format_prompt(dataset: Dataset):
            prompts = []
            for i in range(len(dataset["prompt"])):
                prompt = PROMPT_TEMPLATE.format(
                    QUESTION=dataset["prompt"][i], ANSWER=dataset["chosen"][i]
                )
                prompts.append(prompt)
            return prompts

        instruction_template_ids = tokenizer.encode(
            INSTRUCTION_TEMPLATE, add_special_tokens=False
        )[2:]
        response_template_ids = tokenizer.encode(
            ANSWER_TEMPLATE, add_special_tokens=False
        )[2:]

        data_collator = DataCollatorForCompletionOnlyLM(
            instruction_template=instruction_template_ids,
            response_template=response_template_ids,
            tokenizer=tokenizer,
        )

        trainer = HFSFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            args=self.training_args,
            formatting_func=format_prompt,
            data_collator=data_collator,
            peft_config=peft_config,
        )
        trainer.train()

        _model_name = model.config.name_or_path.split("/")[-1]
        path = os.path.join(self.training_args.output_dir, _model_name)
        trainer.save_model(path)

        return trainer.model
