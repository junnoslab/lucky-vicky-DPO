from typing import Optional
import logging

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    # General
    logger_level: Optional[int] = field(
        default=logging.INFO, metadata={"help": "logger level"}
    )

    # Model
    model_name_or_path: Optional[str] = field(
        default="MLP-KTLim/llama-3-Korean-Bllossom-8B",
        metadata={"help": "the location of the SFT model name or path"},
    )
    rank: Optional[int] = field(
        default=32, metadata={"help": "Lora attention dimension (the “rank”)"}
    )
    lora_alpha: Optional[float] = field(
        default=32, metadata={"help": "alpha parameter for Lora scaling"}
    )
    lora_dropout: Optional[float] = field(
        default=0.1, metadata={"help": "dropout probability for Lora layers"}
    )
    bias: Optional[str] = field(
        default="lora_only",
        metadata={
            "help": "bias type for LoRA. Can be ‘none’, ‘all’ or ‘lora_only’. If ‘all’ or ‘lora_only’, the corresponding biases will be updated during training"
        },
    )

    # Dataset
    dataset_name: Optional[str] = field(
        default="maywell/ko_Ultrafeedback_binarized",
        metadata={"help": "name of the dataset"},
    )

    _eval_ratio: Optional[float] = field(
        default=0.2, metadata={"help": "slice percentage for eval dataset"}
    )

    @property
    def eval_ratio(self) -> Optional[float]:
        return self._eval_ratio

    @eval_ratio.setter
    def eval_ratio(self, value: Optional[float]):
        if value is not None and (value < 0.0 or value > 1.0):
            raise ValueError("eval_ratio must be between 0.0 and 1.0")
        self._eval_ratio = value

    # Train
    beta: Optional[float] = field(
        default=0.1, metadata={"help": "the beta parameter for DPO loss"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=2, metadata={"help": "train batch size per device"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=2, metadata={"help": "eval batch size per device"}
    )
    learning_rate: Optional[float] = field(
        default=5e-4, metadata={"help": "optimizer learning rate"}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata={"help": "the lr scheduler type"}
    )
    warmup_steps: Optional[int] = field(
        default=50, metadata={"help": "the number of warmup steps"}
    )
    optimizer_type: Optional[str] = field(
        default="paged_adamw_32bit", metadata={"help": "the optimizer type"}
    )
    max_prompt_length: Optional[int] = field(
        default=512, metadata={"help": "the maximum prompt length"}
    )
    max_length: Optional[int] = field(
        default=1024, metadata={"help": "the maximum sequence length"}
    )
    eval_steps: Optional[int] = field(
        default=300, metadata={"help": "the evaluation frequency"}
    )

    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )

    save_steps: Optional[int] = field(
        default=300, metadata={"help": "the saving frequency"}
    )

    output_dir: Optional[str] = field(
        default="./res", metadata={"help": "the output directory"}
    )
