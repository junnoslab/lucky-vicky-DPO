from enum import StrEnum, unique

from torch import dtype
import torch


@unique
class Models(StrEnum):
    """Enum for model names."""

    BLOSSOM_8B = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    EEVE_10_8B = "yanolja/EEVE-Korean-10.8B-v1.0"

    @property
    def dtype(self) -> dtype:
        if self == Models.BLOSSOM_8B:
            return torch.float16
        if self == Models.EEVE_10_8B:
            return torch.bfloat16
        else:
            return torch.float32

    @property
    def name(self) -> str:
        return self.value
