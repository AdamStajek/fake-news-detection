from abc import ABC, abstractmethod


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

