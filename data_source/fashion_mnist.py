import gzip

import numpy as np
import torch
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset


class FashionMNIST(Dataset):
    """Dataset for loading Fashion MNIST from disk.
    """

    def __init__(self, img_path, label_path, autoencoder_mode=False):
        """

        :param img_path: path to image dataset file (assumed to be in .gz format)
        :param label_path: path to label file (assumed to be in .gz format)
        :param autoencoder_mode:  return inputs as labels if true.
        """

        # treat the paths as relative to the original working dir, not the hydra dir
        img_path, label_path = to_absolute_path(img_path), to_absolute_path(label_path)

        self.autoencoder_mode = autoencoder_mode
        self.images, self.labels = self.load_mnist(img_path, label_path)
        self.images = torch.as_tensor(self.images, dtype=torch.float) / 255.0
        self.labels = torch.as_tensor(self.labels, dtype=torch.long)

    def __getitem__(self, index):
        if self.autoencoder_mode:
            return self.images[index], self.images[index]
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)

    @staticmethod
    def load_mnist(img_path, label_path):
        with gzip.open(label_path, 'rb') as lbl_path:
            labels = np.frombuffer(lbl_path.read(), dtype=np.uint8,
                                   offset=8)

        with gzip.open(img_path, 'rb') as img_path:
            images = np.frombuffer(img_path.read(), dtype=np.uint8,
                                   offset=16).reshape(len(labels), 784)

        # create writeable copy
        return np.array(images), np.array(labels)
