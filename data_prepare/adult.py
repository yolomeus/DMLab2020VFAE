import os
from urllib import parse, request

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_prepare.base_prepare import DatasetPrepare


class AdultPrepare(DatasetPrepare):
    """Downloads and pre-processes the Adult dataset."""

    def __init__(self, target_dir, download_dir, base_url=None, n_buckets=5, n_splits=5, val_split=0.3):
        """

        :param target_dir: directory to put processed data
        :param download_dir: directory for storing the downloaded raw data
        :param base_url: base url to download the adult data files from
        :param n_buckets: number of buckets for binary quantization of continuous features
        :param n_splits: number of different train - validation splits to generate
        :param val_split: percentage size of the validation split between (0, 1)
        """
        super().__init__(target_dir, download_dir)

        default_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/'
        self.base_url = default_url if base_url is None else base_url
        self.file_names = ['adult.' + name for name in ['data', 'test', 'names']]

        self.n_buckets = n_buckets
        self.n_splits = n_splits
        self.val_split = val_split
        self.col_names = ['age',
                          'workclass',
                          'fnlwgt',
                          'education',
                          'education-num',
                          'marital-status',
                          'occupation',
                          'relationship',
                          'race',
                          'sex',
                          'capital-gain',
                          'capital-loss',
                          'hours-per-week',
                          'native-country',
                          'label']

    def download(self):
        for filename in self.file_names:
            src_url = parse.urljoin(self.base_url, filename)
            download_uri = os.path.join(self.download_dir, filename)
            print(f'downloading to {download_uri}...')
            request.urlretrieve(src_url, download_uri)

    def prepare(self):
        train_file = os.path.join(self.download_dir, 'adult.data')
        test_file = os.path.join(self.download_dir, 'adult.test')

        x_train = pd.read_csv(train_file, sep=',', names=self.col_names)
        x_test = pd.read_csv(test_file, sep=',', names=self.col_names, skiprows=[0])
        self._clean(x_train)
        self._clean(x_test)

        train_len = len(x_train)
        # preprocess both ds as one block
        x = pd.concat([x_train, x_test])

        # encode labels
        y = x.pop('label')
        y = y.apply(lambda text: text.strip().strip('.'))
        y[y == '>50K'] = 1
        y[y == '<=50K'] = 0

        # separate categorical and numerical attributes
        cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                     'native-country']

        cat_features = x[cat_names]
        numeric_features = x.drop(columns=cat_names)

        # binarize sensitive variable
        age = numeric_features.pop('age')
        age[age < 65] = 0
        age[age >= 65] = 1

        # onehot encode
        onehot_cat = pd.get_dummies(cat_features)
        onehot_num = [pd.cut(numeric_features[name], self.n_buckets).astype(str) for name in numeric_features]
        onehot_num = pd.concat([pd.get_dummies(series, prefix=series.name) for series in onehot_num], axis=1)

        x_onehot = pd.concat([onehot_cat, onehot_num], axis=1)
        x_onehot['age_>=65'] = age
        x_onehot['label'] = y
        onehot_train, onehot_test = x_onehot[:train_len], x_onehot[train_len:]

        # following Zemel et al. (2013) as suggested in the paper, we generate and store multiple validation splits
        for i in range(self.n_splits):
            train_split, val_split = train_test_split(onehot_train, test_size=self.val_split)

            # save to disk
            split_dir = f'split_{i}'
            os.makedirs(os.path.join(self.target_dir, split_dir), exist_ok=True)
            train_file = os.path.join(self.target_dir, split_dir, 'adult_train.pkl')
            val_file = os.path.join(self.target_dir, split_dir, 'adult_validation.pkl')

            train_split.to_pickle(train_file)
            val_split.to_pickle(val_file)

        test_file = os.path.join(self.target_dir, 'adult_test.pkl')
        onehot_test.to_pickle(test_file)

    @staticmethod
    def _clean(dataset):
        dataset.replace(' ?', np.nan, inplace=True)
        dataset.dropna(inplace=True)
        dataset.drop(columns='fnlwgt', inplace=True)
        dataset.drop_duplicates(inplace=True)
