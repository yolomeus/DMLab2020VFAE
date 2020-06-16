import pandas as pd
import torch
from hydra.utils import to_absolute_path

from torch.utils.data import Dataset


class Adult(Dataset):
    """Dataset class for the adult dataset. Loads the pandas dataframe from disk and uses it in-memory."""

    def __init__(self, data_path, predict_s=False):
        """

        :param data_path:
        :param predict_s: predict the sensitive variable from the data instead of the labels.
        """
        data_path = to_absolute_path(data_path)
        self.ds = pd.read_pickle(data_path)
        self.predict_s = predict_s

    def __getitem__(self, index):
        item = self.ds.iloc[index]
        if not self.predict_s:
            y = torch.as_tensor(item['label'], dtype=torch.float32).unsqueeze(-1)
            x = item.drop('label').to_numpy().astype('float32')
        else:
            y = torch.as_tensor(item['age_>=65'], dtype=torch.float32).unsqueeze(-1)
            x = item.drop(index=['age_>=65', 'label']).to_numpy().astype('float32')

        return x, y

    def __len__(self):
        return len(self.ds)
