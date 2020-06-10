import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader


class LightningModel(LightningModule):
    """General model wrapper for training pytorch models using the pytorch-lightning library. This class is responsible
    for configuring the whole training and evaluation process.
    """

    def __init__(self, hparams: DictConfig):
        """
        :param hparams: contains model hyperparameters and training settings.
        """
        super().__init__()

        # when loading from a PL checkpoint, hparams is a dict
        if not isinstance(hparams, DictConfig):
            hparams = DictConfig(hparams)
        self.hparams = OmegaConf.to_container(hparams, resolve=True)

        self.model = instantiate(hparams.model)
        self.loss = instantiate(hparams.loss)
        # we pass the model parameters to the optimizer's constructor
        self.optimizer = instantiate(hparams.optimizer, self.model.parameters())

        self.metrics = [instantiate(metric) for metric in hparams.metrics]

        self.dataset_conf = hparams.dataset
        self.train_conf = hparams.training
        self.test_conf = hparams.testing

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.model(x)
        loss = self.loss(y_pred, y_true)

        logs = {'batch_loss': loss}
        return {'loss': loss, 'log': logs, 'y_pred': y_pred, 'y_true': y_true}

    def validation_step(self, batch, batch_idx):
        x, y_true = batch
        y_pred = self.model(x)
        return {'y_pred': y_pred, 'y_true': y_true}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def training_epoch_end(self, outputs):
        return self._epoch_end('train', outputs)

    def validation_epoch_end(self, outputs):
        return self._epoch_end('val', outputs)

    def test_epoch_end(self, outputs):
        return self._epoch_end('test', outputs)

    def _epoch_end(self, prefix, outputs):
        """Compute loss and all metrics at the end of an epoch.

        :param prefix: prefix for logs e.g. train, test, validation
        :param outputs: gathered outputs from *_epoch_end
        :return: a dict containing loss and metric logs.
        """
        y_pred, y_true = self._unpack_outputs('y_pred', outputs), self._unpack_outputs('y_true', outputs)

        logs = {f'{prefix}_' + self._classname(metric): metric(y_pred, y_true) for metric in self.metrics}
        loss = self.loss(y_pred, y_true)
        # when testing we want to log a scalar and not a tensor
        if prefix == 'test':
            loss = loss.item()
        logs[f'{prefix}_loss'] = loss

        return {'log': logs}

    def train_dataloader(self):
        train_ds = instantiate(self.dataset_conf.train)
        train_dl = DataLoader(train_ds,
                              self.train_conf.batch_size,
                              shuffle=True,
                              num_workers=self.hparams['num_workers'])
        return train_dl

    def val_dataloader(self):
        train_ds = instantiate(self.dataset_conf.validation)
        val_dl = DataLoader(train_ds,
                            self.test_conf.batch_size,
                            num_workers=self.hparams['num_workers'])
        return val_dl

    def test_dataloader(self):
        test_conf = self.test_conf
        train_ds = instantiate(self.dataset_conf.test)
        test_dl = DataLoader(train_ds,
                             test_conf.batch_size,
                             num_workers=self.hparams['num_workers'])
        return test_dl

    def configure_optimizers(self):
        return self.optimizer

    @staticmethod
    def _unpack_outputs(key, outputs):
        """Get the values of each output dict at key.

        :param key: key that gets the values from each output dict.
        :param outputs: a list of output dicts.
        :return: the concatenation of all output dict values at key.
        """
        return torch.cat(list(map(lambda x: x[key], outputs)))

    @staticmethod
    def _classname(obj, lower=True):
        """Get the classname of an object.

        :param obj: any python object.
        :param lower: return the name in lowercase.
        :return: the classname as string.
        """
        name = obj.__class__.__name__
        return name.lower() if lower else name
