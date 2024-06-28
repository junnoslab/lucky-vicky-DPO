from enum import StrEnum, unique


@unique
class Models(StrEnum):
    """Enum for model names."""

    BLOSSOM = "MLP-KTLim/llama-3-Korean-Bllossom-8B"
