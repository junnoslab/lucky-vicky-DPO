import logging

from datasets import Dataset, DatasetDict, load_dataset as load_hf_dataset

from .datasets import Datasets

_LOGGER = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, eval_ratio: float = 0.1, test_ratio: float = 0.1):
        self.train_ratio = 1.0 - eval_ratio - test_ratio
        self.eval_ratio = eval_ratio
        self.test_ratio = test_ratio

    def load_dataset(self, dataset: Datasets) -> DatasetDict | Dataset:
        _path_components = dataset.path.split(".")
        _LOGGER.info(f"Loading dataset: {dataset}")

        if len(_path_components) == 1:  # HF ID
            _dataset = load_hf_dataset(dataset.value)
            return _dataset
        elif len(_path_components) == 2:  # Local path
            _ext = _path_components[-1]
            _dataset = self.__load_dataset_from_local(dataset, ext=_ext)
            return _dataset
        else:
            raise ValueError(f"Invalid dataset path: {dataset}")

    def __load_dataset_from_local(self, dataset: Datasets, ext: str) -> Dataset:
        if ext == "parquet":
            _dataset = Dataset.from_parquet(dataset.path)
            if self.train_ratio == 1.0:
                return DatasetDict({"train": _dataset})
            return self.__split(
                _dataset,
                split_ratio=(self.train_ratio, self.eval_ratio, self.test_ratio),
            )

    def __split(
        self,
        dataset: DatasetDict,
        split_ratio: tuple[float, float, float],
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
