from abc import ABC, abstractmethod
from typing import Literal

import pandas as pd


class FakeNewsDataset(ABC):
    """An abstract base class for all fake news datasets used in a project."""

    @abstractmethod
    def __len__(self) -> int:
        """Get a length of a dataset.

        Raises:
            NotImplementedError: An errors if user didn't implement it in child class

        Returns:
            int: The length of a dataset

        """
        msg = "You must implement the __len__ method!"
        raise NotImplementedError(msg)

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[str, int]:
        """Get an item from the dataset.

        Args:
            idx (int): Index of the item to retrievee

        Raises:
            NotImplementedError: An errors if user didn't implement it in child class

        Returns:
            tuple[str, int]: A tuple containing the statement and its label

        """
        msg = "You must implement the __getitem__ method!"
        raise NotImplementedError(msg)

    def _split_dataset(self, dataset: pd.DataFrame, split:
                       Literal["train", "validation"]) -> pd.DataFrame:
        if split == "train":
            dataset = dataset.iloc[:int(0.8 * len(dataset))]
        elif split == "validation":
            dataset = dataset.iloc[int(0.8 * len(dataset)):]
        return dataset

