import torch

from simplepytorch import api
from simplepytorch import datasets as D


class LetsTrainSomething(api.FeedForwardModelConfig):
    """An example configuration that pulls together all the components needed
    to train a model.

    For extra features and documentation, consult the
    api.FeedForwardModelConfig class, and also explore the library
    available under simplepytorch.api.*
    """
    run_id = 'dev'  # name of the experiment.

    epochs = 60
    batch_size = 10

    # define your own parameters or configuration
    my_custom_parameter = 123  # shows up on command-line
    # also this way helps organize the arguments better:
    __groupname_params = api.CmdlineOptions(
        'groupname', {'argname': 'default_value', 'someinteger': 123})
    # override the defaults on command-line voa --groupname-argname somevalue
    # and access variables in code using self.groupname_argname

    def get_lossfn(self):
        def loss_binary_cross_entropy(input, target):
            batch_size = target.shape[0]
            return torch.nn.functional.binary_cross_entropy(
                input.view(batch_size, -1), target.view(batch_size, -1))
        return loss_binary_cross_entropy

    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters())

    def get_datasets(self):
        return super().get_datasets({
            'rite_train', D.RITE(use_train_set=True),
            'rite_test', D.RITE(use_train_set=False)})

    def get_data_loaders(self):
        return super().get_data_loaders({
            'train': torch.utils.data.DataLoader(self.datasets.rite_train),
            #  'test': torch.utils.data.DataLoader(self.datasets.rite_test),
        })

    def get_model(self):
        return torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=3, out_channels=3,
            init_features=128, pretrained=False)
    _FeedForwardModelConfig__model_params = None

    def log_minibatch(self, batch_idx, X, y, yhat, loss):
        batch_size = y.shape[0]
        self.epoch_cache.add(
            'train_loss', loss.item())
        self.epoch_cache.streaming_mean(
            'train_confusion_matrix', torch.tensor([[4,2], [1,3]]), batch_size)

    def log_epoch(self):
        return super().log_epoch({
            'train_loss': self.epoch_cache['train_loss'],
            'train_acc': api.confusion_matrix_stats.matthews_correlation_coeff(
                self.epoch_cache['train_confusion_matrix']),
            'test_loss': 123
        })

    def get_log_header(self):
        super().get_log_header(['train_loss', 'train_acc', 'test_loss'])
