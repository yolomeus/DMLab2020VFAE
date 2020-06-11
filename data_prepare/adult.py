import os
from urllib import parse, request

from data_prepare.base_prepare import DatasetPrepare


class AdultPrepare(DatasetPrepare):
    """Downloads and pre-processes the Adult dataset."""

    def __init__(self, target_dir, download_dir, base_url=None):
        super().__init__(target_dir, download_dir)
        default_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/adult/'
        self.base_url = default_url if base_url is None else base_url
        self.file_names = ['adult.' + name for name in ['data', 'test', 'names']]

    def download(self):
        for filename in self.file_names:
            src_url = parse.urljoin(self.base_url, filename)
            download_uri = os.path.join(self.download_dir, filename)
            print(f'downloading to {download_uri}...')
            request.urlretrieve(src_url, download_uri)

    def prepare(self):
        pass
        # TODO
