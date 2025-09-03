from typing import Literal
import datasets
from custom_datasets.fake_news_dataset import FakeNewsDataset

class LiarLoader(FakeNewsDataset):
    """Liar Dataset Loader - Loads the Liar Dataset (https://aclanthology.org/P17-2067/) 
    from the datasets library.

    The Liar dataset contains political statements labeled for truthfulness.
    Each instance includes the statement text, its label, and various metadata
    such as speaker information, subject, and context.

    Args:
        split (Literal["train", "test", "validation"]): The split of the dataset to load.
            Defaults to "train".

    Attributes:
        statements (list): List of political statements
        labels (list): List of truthfulness labels (0: false, 1: half-true, 2: mostly-true, 3: true, 4: mostly-false, 5: pants-fire)
        metadata (dict): Dictionary containing additional features: 
        'id', 'subject', 'speaker', 
        'job_title', 'state_info', 'party_affiliation', 
        'barely_true_counts', 'false_counts', 
        'half_true_counts', 'mostly_true_counts', 
        'pants_on_fire_counts', 'context'
    """
    def __init__(self, split: Literal["train", "test", "validation"] = "train"):
        self.dataset = datasets.load_dataset("ucsbnlp/liar", split=split)
        self.statements = self.dataset["statement"]
        self.labels = self.dataset["label"]
      
        excluded_features = ["statement", "label"]
        self.metadata = {
            feature: self.dataset[feature]
            for feature in self.dataset.column_names
            if feature not in excluded_features
        }

        self.id2label = {0: "false", 1: "half-true", 2: "mostly-true", 3: "true", 4: "mostly-false", 5: "pants-fire"}
        self.label2id = {v: k for k, v in self.id2label.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.statements[idx], self.labels[idx]
    
