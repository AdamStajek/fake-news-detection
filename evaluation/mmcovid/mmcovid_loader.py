from pathlib import Path
from typing import Literal

import pandas as pd

from evaluation.fake_news_dataset import FakeNewsDataset

DATASET_PATH = Path(__file__).parent.parent / "data" / "mmcovid" / "english_news.csv"


class MMCovidLoader(FakeNewsDataset):
    """MMCovid Dataset Loader - Loads the MMCovid Fake News Dataset.

    The MMCovid dataset contains claims about Covid19.
    """

    def __init__(
        self,
        n: int,
        split: Literal["train", "validation"] = "train",
        *,
        random: bool = True,
    ) -> None:
        """Create an MMCovidLoader instance.

        Args:
            n: Number of samples to load.
            split: A train or validation split for the dataset.
                Defaults to "train".
            random: Whether to sample randomly or take the first n samples.

        """
        dataset = pd.read_csv(DATASET_PATH)[["claim", "label"]]
        if random:
            dataset = dataset.sample(n=n, random_state=42).reset_index(drop=True)
        else:
            dataset = dataset.head(n)
        self.dataset = self._split_dataset(dataset, split)

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: Length of the dataset.

        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        """Get an item from the dataset by index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple[str, int]: A tuple containing the claim text and its label.

        """
        return self.dataset.iloc[idx]["claim"], self.dataset.iloc[idx]["label"]

