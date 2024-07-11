import logging
import os

from datasets import Dataset, DatasetDict
from peft.peft_model import PeftModel
from peft.config import PeftConfig
from transformers import (
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from trl import DPOConfig, DPOTrainer as HFDPOTrainer

from ..utils import TrainConfig
from ..utils.constants import TRAIN_ADAPTER_NAME, REFERENCE_ADAPTER_NAME
from ..utils.secrets import HUGGINGFACE_HUB_TOKEN
from ..utils.templates import PROMPT_TEMPLATE

_LOGGER = logging.getLogger(__name__)


class DPOTrainer:
    training_args: DPOConfig

    def __init__(self, config: TrainConfig):
        self.training_args = DPOConfig(
            output_dir=config.output_dir,
            max_length=2048,
            max_prompt_length=1024,
            num_train_epochs=config.epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            beta=0.1,
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
            save_total_limit=config.save_total_limit,
            save_steps=config.save_steps,
            logging_strategy="steps",
            logging_steps=config.logging_steps,
            push_to_hub=config.push_to_hub is not None,
            hub_model_id=config.push_to_hub,
            hub_token=HUGGINGFACE_HUB_TOKEN,
        )

    def train(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset,
        peft_config: PeftConfig,
    ) -> PreTrainedModel:
        _model_name = model.config.name_or_path.split("/")[-1]
        _model_result_path = os.path.join(self.training_args.output_dir, _model_name)

        # Setup tokenizer
        tokenizer.padding_side = "right"

        # Load the adapter.
        _LOGGER.info(f"Using PEFT with config: {peft_config}")
        _model = PeftModel.from_pretrained(
            model,
            model_id=_model_result_path,
            adapter_name=TRAIN_ADAPTER_NAME,
            is_trainable=True,
            config=peft_config,
        )
        del model

        # Load the second adapter, but with a different name.
        _model.load_adapter(
            model_id=_model_result_path,
            adapter_name=REFERENCE_ADAPTER_NAME,
            is_trainable=True,
            peft_config=peft_config,
        )

        # Setup adapter
        self.training_args.model_adapter_name = TRAIN_ADAPTER_NAME
        self.training_args.ref_adapter_name = REFERENCE_ADAPTER_NAME

        def format_prompt(dataset: Dataset) -> DatasetDict:
            return DatasetDict(
                {
                    "index": dataset["index"],
                    "prompt": [
                        PROMPT_TEMPLATE.format(QUESTION=input, ANSWER="")
                        for input in dataset["prompt"]
                    ],
                    "chosen": dataset["chosen"],
                    "rejected": dataset["rejected"],
                }
            )

        dataset = dataset["train"].map(
            format_prompt,
            batched=True,
            remove_columns=["index", "prompt", "chosen", "rejected"],
        )

        trainer = HFDPOTrainer(
            model=_model,
            train_dataset=dataset,
            tokenizer=tokenizer,
            args=self.training_args,
        )
        trainer.train()

        trainer.save_model(_model_result_path)
        _model.save_adapter(_model_result_path)

        return trainer.model
