"""
Config and functions to train and test feedforward networks using backprop
"""
from collections import namedtuple
import time
from efficientnet_pytorch import EfficientNet
from os.path import join
import os
import abc
import logging
import pretrainedmodels
import numpy as np
import random
import torch
import torch.optim

from .logging_tools import LogRotate, CsvLogger
from .cache import Cache
from .cmdline import CmdlineOptions, ModelConfigABC


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def train_one_epoch(config):
    config.model.train()
    for batch_idx, (X, y) in enumerate(config.data_loaders.train):
        X, y = X.to(config.device), y.to(config.device)
        config.optimizer.zero_grad()
        yhat = config.model(X)
        loss = config.lossfn(yhat, y)
        loss.backward()
        config.optimizer.step()
        with torch.no_grad():
            config.log_minibatch(
                batch_idx=batch_idx, X=X, y=y, yhat=yhat, loss=loss)


def train(config):
    for epoch in range(config.cur_epoch + 1, config.epochs + 1):
        config.cur_epoch = epoch
        config.epoch_cache.clear()

        config.train_one_epoch()
        with torch.no_grad():
            config.log_epoch()
            config.save_checkpoint()
        if config.eval_early_stopping():
            log.info("Early Stopping condition activated")
            break


class FeedForwardModelConfig(ModelConfigABC):
    run_id = str

    batch_size = int
    epochs = int
    base_dir = './data'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @abc.abstractmethod
    def get_lossfn(self):
        # return torch.nn.BCELoss()
        raise NotImplementedError("Your implementation here")

    @abc.abstractmethod
    def get_optimizer(self):
        #  return torch.optim.Adam(
        #      self.model.parameters(), **self.__optimizer_params.kwargs(self))
        raise NotImplementedError("Your implementation here")

    @abc.abstractmethod
    def get_datasets(self, datasets=None):
        """Convenience function.  Not necessary, but could be helpful
        to implement this method if you have data loaders or configuration that
        change depending on the dataset.  Example implementation:

        def get_datasets(self):
            return super().get_datasets({
                'myimagenet': torch.utils.data.Dataset(),
                'myotherset': torch.utils.data.Dataset(),
                'mytestdata': torch.utils.data.Dataset()})

        Then, it's available to your class methods as, for example:
            self.datasets.myimagenet
        """
        if datasets is not None:
            return namedtuple('datasets', datasets.keys())(**datasets)

    @abc.abstractmethod
    def get_data_loaders(self, loaders):
        """Implement your data loaders.  If you use the default
        `train_one_epoch` function, you will need a 'train' loader.  Any other
        loaders you may define depend on how you use them.

        def get_data_loaders(self):
            return super().get_data_loaders({
                'train': torch.utils.data.DataLoader(self.datasets.myimagenet)
                'val': torch.utils.data.DataLoader(self.datasets.myotherset)
                'test': torch.utils.data.DataLoader(self.datasets.mytestdata)
                })
        """
        return namedtuple('data_loaders', loaders.keys())(**loaders)

    __model_params = CmdlineOptions(
        'model', {'name': str, # --model-name resnet18  (required option)
                  'num_classes': int, # --model-num-classes 1000  (required option)
                  })

    def get_model(self):
        """
        Load a pre-trained model from one of many locations:
            torch.hub
            or torchvision.models
            or github:Cadene/pretrained-models.pytorch
            or efficientnet-pytorch

        You can then pass from command-line:

            --model-name hub:pytorch/vision:deeplabv3_resnet50
            --model-name efficientnet-b0

        Or you can set in your configuration class:

            class MyConfig(FeedForwardModelConfig):
                __model_params = api.CmdlineOptions(
                    'model', {'name': 'efficientnet-b0', 'num_classes': 4})

        You can also just ignore and override this method to define your own model:

            class A(FeedForwardModelConfig):
                # unset existing CmdlineOptions if you won't used them
                _FeedForwardModelConfig__model_params = None

                def get_model(self):
                    mymodel = torch.nn.Module()
                    return mymodel

        """
        if self.model_name.startswith('efficientnet-b'):
            model = EfficientNet.from_pretrained(
                self.model_name, num_classes=self.model_num_classes)
        elif self.model_name.startswith('hub:'):
            github_repo, model_name = self.model_name.split(':')[1:]
            model = torch.hub.load(github_repo, model_name, pretrained=True)
        elif self.model_name.startswith('cadene:'):
            model = getattr(pretrainedmodels, self.model_name.split(':', 1)[1])()
            if self.model_num_classes != model.last_linear.out_features:
                model.last_linear = torch.nn.Linear(
                    model.last_linear.in_features, self.model_num_classes)
        else:
            raise Exception("Don't know how to load model: %s" % self.model_name)
        return model

    def train(self):
        """Train the model.  If want to override this method, it's probably
        better to just start from scratch.
        """
        log.info('training model')
        return train(self)

    def train_one_epoch(self):
        """Train model for one epoch.  This method is expected to call
        log_minibatch.  Subclasses can override this to define custom behavior.
        """
        return train_one_epoch(self)

    def eval_early_stopping(self):
        """Subclasses can override this to implement early stopping behavior.
        It is called at the end of every epoch."""
        return False

    def log_minibatch(self, extra_log_data=None, *,
                      batch_idx, X=None, y=None, yhat=None, loss=None):
        """You can override this method to log something for each minibatch.

        It is expected that `train_one_epoch` will call this function for each
        minibatch.

        Add your extra_log_data like below.  You can also
        store or aggregate metrics in a cache that may be useful for the
        `log_epoch` function:

            def log_minibatch(self, batch_idx, X, y, yhat, loss):
                extra_log_data = {'train_loss': loss.item()}
                super().log_minibatch(extra_log_data, batch_idx=batch_idx)

                # you can also store cumulative data for the log_epoch function
                self.epoch_cache.streaming_mean(
                    'train_loss', loss.item(), batch_size),
                self.epoch_cache.streaming_mean('train_acc', .5, batch_size)
        """
        intv = self.log_minibatch_interval
        if intv > 0 and batch_idx % intv == intv - 1:
            log_data = {"epoch": self.cur_epoch, "batch_idx": batch_idx}
            if extra_log_data is not None:
                log_data.update(extra_log_data)

            self.logwriter.writerow(log_data, ignore_missing=True)

    def log_epoch(self, extra_log_data=None):
        """You can override this method to log something at the end of each epoch.

        It is expected that `train` calls this function.
        You can store cumulative stats needed here in self.epoch_cache,
        (the cache is cleared at the start of each epoch by `train`).

        A simple example of overriding this method, where we assume the
        epoch_cache contains data you previously saved (ie see `log_minibatch`)

            def log_epoch(self):
                extra_log_data = {
                    'train_loss': self.epoch_cache['train_loss'].mean,
                    'train_acc': self.epoch_cache['train_acc'].mean},
                    'val_acc': 0.51}
                super().log_epoch(extra_log_data)
        """
        intv = self.log_epoch_interval
        if intv > 0 and self.cur_epoch % intv == intv - 1:
            log_data = dict(batch_idx=None, epoch=self.cur_epoch, timestamp=int(time.time()))
            if extra_log_data is not None:
                log_data.update(extra_log_data)

            self.logwriter.writerow(log_data, ignore_missing=True)
            self.logwriter.flush()

    log_epoch_interval = 1  # log perf metrics after every N epochs
    log_minibatch_interval = 10  # log train perf after every Nth batch_idx

    def get_log_filepath(self):
        return f'{self.base_dir}/results/{self.run_id}/perf.csv'

    def get_log_header(self, extra_fields=()):
        """Specify which fields to save to csv file each epoch and minibatch
        Subclasses should define this if they have logic in log_epoch:

        def get_log_header(self):
            return super().get_log_header(['train_loss', 'val_loss', ...])
        """
        rv = ['timestamp', 'epoch', 'batch_idx']
        rv.extend(extra_fields)
        return rv

    __checkpoint_params = CmdlineOptions(
        'checkpoint', {
            'interval': 1,  # --checkpoint-interval 1
            'fp': '{self.base_dir}/results/{self.run_id}/model_checkpoints/{self.checkpoint_fname}',
            'fname': 'epoch_{self.cur_epoch}.pth'
        })

    # cur_epoch is updated as model trains and used to load checkpoint.
    # the epoch number is actually 1 indexed.  By default, try to load
    # the epoch 0 file, which won't exist unless you manually put it
    # there.
    cur_epoch = 0

    def load_checkpoint(self):
        """Logic to restore from a checkpoint.  Should be a
        corresponding implementation in get_checkpoint_state.
        Should do nothing if checkpoint file does not exist.

        the __init__ function carefully handles initialization of random state
        """
        fp = self.checkpoint_fp.format(self=self).format(self=self)
        if not os.path.exists(fp):
            return {}
        log.info("Restoring from checkpoint: %s", fp)
        checkpoint = torch.load(fp, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.optimizer.load_state_dict(
            checkpoint['optimizer_state_dict'])
        self._randomness_init = checkpoint['randomness_init']
        self.cur_epoch = checkpoint['cur_epoch']
        return checkpoint

    def save_checkpoint(self, force_save=False):
        """Override this to control logic for when checkpoints are saved.
        Called once per epoch.  Example implementation:

            def save_checkpoint(self):
                if self.is_best_performing_epoch():
                    super().save_checkpoint(force_save=True)
        """
        intv = self.checkpoint_interval
        if force_save or (intv > 0 and self.cur_epoch % intv == 0):
            save_fp = self.checkpoint_fp.format(self=self).format(self=self)
            log.info("Save checkpoint %s", save_fp)
            os.makedirs(os.path.dirname(save_fp), exist_ok=True)
            torch.save(self.get_checkpoint_state(), save_fp)

    def get_checkpoint_state(self):
        """State to save in the checkpoint file.  Should be a
        corresponding implementation in load_checkpoint."""
        self._randomness_init.update({
            'np_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
            'torch_cuda_random_state': torch.cuda.get_rng_state_all()})
        return {'cur_epoch': self.cur_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'randomness_init': self._randomness_init}

    def run(self):
        """Start the main loop (ie train model or evaluate on test set).
        Fixed to train by default.

        You can override this in your config class like:
            def run(self):
                if self.training_mode:
                    self.train()  # given by FeedForward
                else:
                    self.test()  # you'd need to create this.
        """
        self.train()

    def _set_random_seeds(self, seeds=np.random.randint(0, 2**32, 3)):
        torch.manual_seed(seeds[0])
        np.random.seed(seeds[1])
        random.seed(seeds[2])

    def __init__(self, config_override_dict):
        self.__dict__.update({k: v for k, v in config_override_dict.items()
                              if v is not None})
        assert isinstance(self.run_id, str), \
            "must define a run_id for this experiment, ie via --run-id testrun"
        os.environ.setdefault('TORCH_HOME', join(self.base_dir, 'torch'))
        if 'cuda:' in self.device:
            torch.cuda.set_device(self.device)

        self.model = self.get_model()
        self.optimizer = self.get_optimizer()

        self._randomness_init = { 'seeds': np.random.randint(0, 2**32, 3) }
        self.load_checkpoint()  # may update _randomness_init

        seeds = self._randomness_init['seeds']
        self._set_random_seeds(seeds)

        self.lossfn = self.get_lossfn()
        self.datasets = self.get_datasets()
        self.data_loaders = self.get_data_loaders()

        self.epoch_cache = Cache()
        self.logwriter = LogRotate(CsvLogger)(self.get_log_filepath(), self.get_log_header())

        # put model on gpu
        self.model.to(self.device)
        # parallelize model if possible
        if self.device == 'cuda' and torch.cuda.device_count() > 1:
            log.info("Using %s GPUs", torch.cuda.device_count())
            self.model = torch.nn.DataParallel(self.model)

        if 'np_random_state' in self._randomness_init:
            # we restored from a checkpoint and need to set the random number
            # generators
            np.random.set_state(self._randomness_init['np_random_state'])
            torch.cuda.set_rng_state_all(
                [x.cpu() for x in self._randomness_init['torch_cuda_random_state']])
            torch.random.set_rng_state(
                self._randomness_init['torch_random_state'].cpu())



    def __repr__(self):
        return "config:%s" % self.run_id


def val_perf_binary_classifier(config, with_logits=False):
    """Return avg loss and accuracy on the validation data.
    Assume validation set data loader is at config.data_loaders.val"""
    config.model.eval()
    totloss = 0
    correct = 0
    N = 0
    with torch.no_grad():
        for X, y in config.data_loaders.val:
            batch_size = X.shape[0]
            X, y = X.to(config.device), y.to(config.device)
            yhat = config.model(X)
            if with_logits:
                yhat = torch.sigmoid(yhat)
            totloss += (config.lossfn(yhat, y.float()) * batch_size).item()
            correct += y.int().eq((yhat.view_as(y) > .5).int()).sum().item()
            N += batch_size
    return {'val_loss': totloss/N, 'val_acc': correct/N}


class FeedForwardBinaryClassifier(FeedForwardModelConfig):
    """Convenience implementation showing how to incorporate train and
    validation accuracy into the logs.  For a binary classifier. Model config
    would inherit from this class instead of FeedForward.
    """
    def log_minibatch(self, extra_log_data=None, *,
                      batch_idx, X, y, yhat, loss):
        num_correct = y.int().eq((yhat.view_as(y) > .5).int()).sum().item()
        batch_size = X.shape[0]
        acc = num_correct / batch_size

        self.epoch_cache.streaming_mean('train_loss', loss.item(), batch_size)
        self.epoch_cache.streaming_mean('train_acc', acc, batch_size)

        super().log_minibatch(
            {"train_loss": loss.item(), 'train_acc': acc}, batch_idx=batch_idx)

    def log_epoch(self, extra_log_data=None, _with_logits=False):
        """Return avg loss and accuracy on the validation data
        if _with_logits=True, apply sigmoid to model output before evaluating
        loss.
        """
        if extra_log_data is None:
            extra_log_data = {}
        extra_log_data.update(val_perf_binary_classifier(self, _with_logits))
        extra_log_data['train_loss'] = self.epoch_cache['train_loss'].mean
        extra_log_data['train_acc'] = self.epoch_cache['train_acc'].mean
        super().log_epoch(extra_log_data)

    def get_log_header(self):
        """Specify which fields are saved to csv file each epoch and batch"""
        rv = super().get_log_header()
        rv.extend(['train_loss', 'train_acc', 'val_loss', 'val_acc'])
        return rv
