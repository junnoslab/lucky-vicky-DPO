from enum import StrEnum, unique

from torch import dtype
import torch


@unique
class Models(StrEnum):
    """Enum for model names."""

    BLOSSOM_8B = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    EEVE_10_8B = "yanolja/EEVE-Korean-10.8B-v1.0"
    EEVE_10_8B_INST = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"

    @classmethod
    def from_value(cls, value) -> "Models":
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"{value} is not a valid {cls.__name__}")

    @property
    def dtype(self) -> dtype:
        if self == Models.BLOSSOM_8B:
            return torch.float16
        elif self == Models.EEVE_10_8B:
            return torch.float16
        elif self == Models.EEVE_10_8B_INST:
            return torch.float16
        else:
            return torch.float32
