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
        self.ds = pd.read_pickle(data_path).drop(columns='sex_ Male')
        self.predict_s = predict_s
        self.protected_var = 'sex_ Female'

    def __getitem__(self, index):
        item = self.ds.iloc[index]
        s = torch.as_tensor(item[self.protected_var], dtype=torch.float32).unsqueeze(-1)
        if not self.predict_s:
            y = torch.as_tensor(item['label'], dtype=torch.float32).unsqueeze(-1)
            x = item.drop('label').to_numpy().astype('float32')
        else:
            y = s
            x = item.drop(index=[self.protected_var, 'label']).to_numpy().astype('float32')

        return x, {'y_true': y, 'is_protected': s}

    def __len__(self):
        return len(self.ds)


class AdultVFAE(Dataset):
    """Adult dataset returning inputs and targets for training the VFAE model."""

    def __init__(self, data_path, predict_s, predict_y):
        data_path = to_absolute_path(data_path)
        self.ds = pd.read_pickle(data_path)
        self.predict_s = predict_s
        self.predict_y = predict_y
        self.protected_var = 'sex_ Female'

    def __getitem__(self, index):
        item = self.ds.iloc[index]
        s = torch.as_tensor(item[self.protected_var], dtype=torch.float32).unsqueeze(-1)
        y = torch.as_tensor(item['label'], dtype=torch.float32).unsqueeze(-1)
        x = item.drop(index=[self.protected_var, 'label']).to_numpy().astype('float32')

        input_dict = {'x': x, 's': s, 'y': y}
        if self.predict_s:
            target_dict = {'y_true': s}
        elif self.predict_y:
            target_dict = {'y_true': y, 'is_protected': s}
        else:
            target_dict = {'x': x, 's': s, 'y': y.squeeze().long()}
        return input_dict, target_dict

    def __len__(self):
        return len(self.ds)
