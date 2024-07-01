from typing import Any

from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase
import torch


class DataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, device: torch.device):
        super().__init__(tokenizer)
        self.device = device

    def __call__(self, features: list[dict[str, Any]]):
        batch = super().__call__(features)
        # Move batch to the specified device
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        return batch
