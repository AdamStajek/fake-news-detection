from pathlib import Path
from typing import Literal

import pandas as pd

from evaluation.fake_news_dataset import FakeNewsDataset

BASE_PATH = Path(__file__).parent / "data" / "isot"
TRUE_PATH = BASE_PATH / "True.csv"
FAKE_PATH = BASE_PATH / "Fake.csv"

class IsotLoader(FakeNewsDataset):
    """ISOT Dataset Loader - Loads the ISOT Fake News Dataset (https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/).

    The ISOT dataset contains news articles labeled as fake or real.
    Each instance includes the article title, text, and its label.

    Args:
        split (Literal["train", "test", "validation"]): The split of the dataset to load
            Defaults to "train".

    Attributes:
        titles (list): List of news article titles
        texts (list): List of news article texts
        labels (list): List of labels (0: fake, 1: real)

    """

    def __init__(self, split: Literal["train", "validation"] = "train") -> None:
        """Create an ISOTLoader instance.

        Args:
            split (Literal["train", "validation"],
            optional): A train or validation split for the dataset. Defaults to "train".

        """
        true_texts = pd.read_csv(TRUE_PATH)[["text"]]
        false_texts = pd.read_csv(FAKE_PATH)[["text"]]
        true_texts, false_texts = self._append_with_label(true_texts, false_texts)
        dataset = pd.concat([true_texts, false_texts], ignore_index=True)
        dataset = dataset.sample(frac=1).reset_index(drop=True)
        self.dataset = self._split_dataset(dataset, split)

    def _append_with_label(self, true_texts: pd.DataFrame, false_texts: pd.DataFrame) \
            -> tuple[pd.DataFrame, pd.DataFrame]:
        true_texts["label"] = 1
        false_texts["label"] = 0
        return true_texts, false_texts

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: Length of the dataset

        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        """Get an item from the dataset by index.

        Args:
            idx (int): Index of the item to retrieve

        Returns:
            tuple[str, int]: A tuple containing the text and its label

        """
        row = self.dataset.iloc[idx]
        text = row["text"]
        label = row["label"]
        return text, label

