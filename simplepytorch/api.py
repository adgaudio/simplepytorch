"""
Code available to model configs, assembled all in one place
"""
import torch.utils.data as TD
import torch.multiprocessing

from . import metrics
from . import datasets
from .cache import Cache
from .cmdline import ModelConfigABC, CmdlineOptions, load_model_config
from .logging_tools import LogRotate, CsvLogger, PickleLogger, HDFLogger
from .early_stopping import EarlyStopping

from .feedforward import (
    FeedForwardModelConfig, FeedForwardBinaryClassifier,
    train, train_one_epoch, val_perf_binary_classifier)


def create_data_loader(config, idxs, dataset, shuffle=True,
                       drop_last=False,
                       num_workers=torch.multiprocessing.cpu_count()-1):
    assert idxs.max() < len(dataset)  # sanity check
    if shuffle:
        sampler = TD.SubsetRandomSampler(idxs)
    else:
        sampler = TD.SequentialSampler(idxs)
        dataset = TD.Subset(dataset, idxs)
    return TD.DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        drop_last=drop_last,
        pin_memory=True, num_workers=num_workers
    )
