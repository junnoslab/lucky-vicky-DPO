from enum import StrEnum, unique

from torch import dtype
import torch


@unique
class Models(StrEnum):
    """Enum for model names."""

    BLOSSOM = "MLP-KTLim/llama-3-Korean-Bllossom-8B"

    @property
    def dtype(self) -> dtype:
        if self == Models.BLOSSOM:
            return torch.float16
