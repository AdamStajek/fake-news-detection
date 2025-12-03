import json
from pathlib import Path
from typing import Literal

import pandas as pd

from evaluation.fake_news_dataset import FakeNewsDataset

BASE_PATH = Path(__file__).parent.parent / "data" / "polish_info"
DATA_PATH = BASE_PATH / "data.json"


class PolishInfoLoader(FakeNewsDataset):
    """Polish Info Dataset Loader.

    Loads the Polish Info fake news dataset.
    The Polish Info dataset contains Polish news statements labeled as
    True, False, or Unclear. Each instance includes the statement text
    and its label.

    Attributes:
        dataset: DataFrame containing statements and labels.

    """

    def __init__(
        self, n: int, split: Literal["train", "validation"] = "train",
    ) -> None:
        """Create a PolishInfoLoader instance.

        Args:
            n: Number of samples to load from the dataset.
            split: A train or validation split for the dataset.
                Defaults to "train".

        """
        with DATA_PATH.open(encoding="utf-8") as f:
            data = json.load(f)

        dataset = pd.DataFrame(data)
        dataset = dataset.sample(
            n=min(n, len(dataset)), random_state=42,
        ).reset_index(drop=True)
        self.dataset = self._split_dataset(dataset, split)

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: Length of the dataset

        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        """Get an item from the dataset by index.

        Args:
            idx (int): Index of the item to retrieve

        Returns:
            tuple[str, str]: A tuple containing the statement text and its label

        """
        row = self.dataset.iloc[idx]
        statement = row["statement"]
        label = row["label"]
        return statement, label

