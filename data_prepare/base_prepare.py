import os
from abc import ABC, abstractmethod


class DatasetPrepare(ABC):
    """Base class for data download and preparation.
    """

    def __init__(self, target_dir, download_dir):
        self.target_dir = target_dir
        self.download_dir = download_dir

        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.target_dir, exist_ok=True)

    @abstractmethod
    def download(self):
        """
        Downloads the dataset.
        """

    @abstractmethod
    def prepare(self):
        """Apply any pre-processing  to the data.
        """
