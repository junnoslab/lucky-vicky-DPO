from typing import Optional
import logging

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # General
    logger_level: Optional[int] = field(
        default=logging.INFO, metadata={"help": "logger level"}
    )
    is_ready_for_training: Optional[bool] = field(
        default=False,
        metadata={
            "help": "whether the model is ready for training so it can be logged in wandb"
        },
    )
    wandb_project_name: Optional[str] = field(
        default="lucky-vicky", metadata={"help": "wandb project name"}
    )

    # Model
    model_name: Optional[str] = field(
        default="yanolja/EEVE-Korean-10.8B-v1.0",
        metadata={"help": "the location of the SFT model name or path"},
    )
    rank: Optional[int] = field(
        default=32, metadata={"help": "Lora attention dimension (the “rank”)"}
    )
    lora_alpha: Optional[float] = field(
        default=64, metadata={"help": "alpha parameter for Lora scaling"}
    )
    lora_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "dropout probability for Lora layers"}
    )
    bias: Optional[str] = field(
        default="none",
        metadata={
            "help": "bias type for LoRA. Can be ‘none’, ‘all’ or ‘lora_only’. If ‘all’ or ‘lora_only’, the corresponding biases will be updated during training"
        },
    )

    # Dataset
    _eval_ratio: Optional[float] = field(
        default=0.0, metadata={"help": "slice percentage for eval dataset"}
    )

    @property
    def eval_ratio(self) -> Optional[float]:
        return self._eval_ratio

    @eval_ratio.setter
    def eval_ratio(self, value: Optional[float]):
        if value is not None and (value < 0.0 or value > 1.0):
            raise ValueError("eval_ratio must be between 0.0 and 1.0")
        self._eval_ratio = value

    _test_ratio: Optional[float] = field(
        default=0.0, metadata={"help": "slice percentage for test dataset"}
    )

    @property
    def test_ratio(self) -> Optional[float]:
        return self._test_ratio

    @test_ratio.setter
    def test_ratio(self, value: Optional[float]):
        if value is not None and (value < 0.0 or value > 1.0):
            raise ValueError("test_ratio must be between 0.0 and 1.0")
        self._test_ratio = value

    dataloader_num_workers: Optional[int] = field(
        default=8, metadata={"help": "the number of dataloader workers"}
    )

    # Train
    _train_mode: Optional[str] = field(
        default="full", metadata={"help": "the training mode (full, sft, dpo)"}
    )

    @property
    def train_mode(self) -> Optional[str]:
        return self._train_mode

    @train_mode.setter
    def train_mode(self, value: Optional[str]):
        if value not in ["full", "sft", "dpo"]:
            raise ValueError("train_mode must be one of 'full', 'sft', 'dpo'")
        self._train_mode = value

    epochs: Optional[int] = field(
        default=3, metadata={"help": "the number of training epochs"}
    )
    beta: Optional[float] = field(
        default=0.1, metadata={"help": "the beta parameter for DPO loss"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=8, metadata={"help": "train batch size per device"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=2, metadata={"help": "eval batch size per device"}
    )
    max_seq_length: Optional[int] = field(
        default=2048, metadata={"help": "the maximum sequence length"}
    )
    learning_rate: Optional[float] = field(
        default=5e-4, metadata={"help": "optimizer learning rate"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata={"help": "the lr scheduler type"}
    )
    warmup_ratio: Optional[float] = field(
        default=0.03, metadata={"help": "the ratio of warmup steps"}
    )
    optimizer_type: Optional[str] = field(
        default="paged_adamw_8bit", metadata={"help": "the optimizer type"}
    )
    eval_steps: Optional[int] = field(
        default=20, metadata={"help": "the evaluation frequency"}
    )

    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )

    save_steps: Optional[int] = field(
        default=20, metadata={"help": "the saving frequency"}
    )

    output_dir: Optional[str] = field(
        default="./res", metadata={"help": "the output directory"}
    )

    save_total_limit: Optional[int] = field(
        default=2, metadata={"help": "the total number of checkpoints to save"}
    )

    save_steps: Optional[int] = field(
        default=2, metadata={"help": "the saving frequency"}
    )

    logging_steps: Optional[int] = field(
        default=2, metadata={"help": "the logging frequency"}
    )

    push_to_hub: Optional[str] = field(
        default="Junnos/lucky-vicky",
        metadata={
            "help": "model id to push to hub. if it's set to None, it won't push to hub"
        },
    )
