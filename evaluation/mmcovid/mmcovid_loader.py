from pathlib import Path
from typing import Literal

import pandas as pd

from evaluation.fake_news_dataset import FakeNewsDataset

DATASET_PATH = Path(__file__).parent / "data" / "mmcovid"

class MMCovidLoader(FakeNewsDataset):
    """MMCovid Dataset Loader - Loads the MMCovid Fake News Dataset(https://github.com/bigheiniu/MM-COVID).

    The MMCovid dataset contains claims about Covid19.
    """

    def __init__(self, n: int, split: Literal["train", "validation"] = "train") -> None:
        """Create an MMCovidLoader instance.

        Args:
            split (Literal["train", "validation"],
            optional): A train or validation split for the dataset. Defaults to "train".

        """
        dataset = pd.read_csv(DATASET_PATH)[["claim", "label"]]
        dataset = dataset[:n]
        self.dataset = self._split_dataset(dataset, split)

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: Length of the dataset.

        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        """Get an item from the dataset by index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple[str, int]: A tuple containing the claim text and its label.

        """
        return self.dataset.iloc[idx]["claim"], self.dataset.iloc[idx]["label"]
