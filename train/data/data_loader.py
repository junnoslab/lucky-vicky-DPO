import logging

from datasets import Dataset, DatasetDict

from .datasets import Datasets

_LOGGER = logging.getLogger(__name__)


class DataLoader:
    def __init__(self) -> None:
        pass

    def load_dataset(self, dataset: Datasets) -> DatasetDict:
        _ext = dataset.path.split(".")[-1]
        _LOGGER.info(f"Loading dataset from {dataset.path}, ext: {_ext}")

        if _ext == "parquet":
            _dataset = Dataset.from_parquet(dataset.path)
            return self.__split(_dataset)
        else:
            raise ValueError("Not supported file.")

    def __split(
        self,
        dataset: DatasetDict,
        split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
    ) -> DatasetDict:
        # Split the dataset into train, eval, and test
        train_valid_dataset = dataset.train_test_split(test_size=1.0 - split_ratio[0])
        valid_test_dataset = train_valid_dataset["test"].train_test_split(
            test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2])
        )
        return DatasetDict(
            {
                "train": train_valid_dataset["train"],
                "eval": valid_test_dataset["train"],
                "test": valid_test_dataset["test"],
            }
        )
