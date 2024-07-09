import os

from enum import StrEnum


class Datasets(StrEnum):
    """Enum for dataset names."""

    LUCKY_VICKY = "Junnos/luckyvicky-DPO"

    @property
    def path(self) -> str:
        dataset_path = os.path.join("data", self.value)
        return dataset_path
