import torch
import torchvision.transforms as tvt

from simplepytorch import api
from simplepytorch import datasets as D


class LetsTrainSomething(api.FeedForwardModelConfig):
    """An example configuration that pulls together all the components needed
    to train a model.

    For extra features and documentation, consult the
    api.FeedForwardModelConfig class, and also explore the library
    available under simplepytorch.api.*

    To use for training, follow the README.  The gist is this:
    # you should have ./data and ./examples in the current directory
    # --> train
    $ simplepytorch ./examples LetsTrainSomething
    $ simplepytorch_debug ./examples LetsTrainSomething
    # --> check results
    $ tail -f ./data/results/dev/perf.csv
    # --> plot results
    $ simplepytorch_plot dev  # dev is the `run_id`
    """

    # options available at the command-line, with default values specified here
    run_id = 'dev'  # name of the experiment.  Used to record results.
    epochs = 60
    # --> define your own parameters or configuration
    my_custom_parameter = 123  # shows up on command-line
    # --> here is a way to organize the arguments better:
    __groupname_params = api.CmdlineOptions(
        'groupname', {'argname': 'default_value', 'someinteger': 123})
    # ... override on command-line via --groupname-argname somevalue
    # ... and access variables in code using self.groupname_argname

    def get_lossfn(self):
        return torch.nn.BCEWithLogitsLoss()

    def get_optimizer(self):
        return torch.optim.Adam(self.model.parameters())

    def get_datasets(self):
        def transform(dct: dict):
            # (just predict the sum of blood vessel pixels)
            return (  # (X, y)
                tvt.functional.to_tensor(dct['image']),
                torch.tensor([tvt.functional.to_tensor(dct['vessel']).sum()]))
        return super().get_datasets({
            'rite_train': D.RITE(use_train_set=True,
                                 getitem_transform=transform),
            #  'rite_val': D.RITE(use_train_set=False,
            #                     getitem_transform=transform)
        })

    def get_data_loaders(self):
        return super().get_data_loaders({
            'train': torch.utils.data.DataLoader(self.datasets.rite_train, batch_size=5),
            #  'test': torch.utils.data.DataLoader(self.datasets.rite_test),
        })

    # get_model
    # --> use a model from torch hub (or some other locations)
    __model_params = api.CmdlineOptions(
        'model', {'name': 'efficientnet-b0', 'num_classes': 1})
    # --> or specify your own custom model
    #  def get_model(self):
    #      return torch.hub.load(
    #          'mateuszbuda/brain-segmentation-pytorch', 'unet',
    #          in_channels=3, out_channels=3,
    #          init_features=128, pretrained=False)
    #  _FeedForwardModelConfig__model_params = None

    def log_minibatch(self, batch_idx, X, y, yhat, loss):
        batch_size = y.shape[0]
        self.epoch_cache.add(
            'train_loss', loss.item())
        self.epoch_cache.streaming_mean(
            'train_mse', ((y.float()-yhat.float())**2).sum().item(), batch_size)
        self.epoch_cache.streaming_mean(
            'some measurement for use at end of epoch',
            yhat.mean().item(), batch_size)

    def log_epoch(self):
        # the returned values populate a log file.
        # you should compute your validation or test set performance here.
        return super().log_epoch({
            'train_loss': self.epoch_cache['train_loss'],
            'train_mse': self.epoch_cache['train_mse'].mean,
            'my measurement': self.epoch_cache[
                'some measurement for use at end of epoch'].mean
        })

    def get_log_header(self):
        return super().get_log_header(
            ['train_loss', 'my measurement', 'train_mse'])

    # consult the api.FeedForwardModelConfig class for more things you can do.
