import pandas as pd
import torch as T
import torchvision.transforms as tvt
import re
import numpy as np
from .glob_image_dir import GlobImageDir


class CheXpert(GlobImageDir):
    """Load CheXpert Dataset, assuming you already have a copy on disk.
    This loads the images and labels under the "train" directory, and applies
    optional transforms.  We also clean up the labels.

    An example usage looks like this:

        >>> dset = CheXpert(
            img_transform=tvt.Compose([
                tvt.RandomCrop((512, 512)),
                tvt.ToTensor(),
            ]),
            getitem_transform=lambda dct: (
                dct['image'],
                CheXpert.format_labels(dct),  # chexpert provided labels
                ]),
            )
        )

        There are 18 classes exposed by the dataset that a model could predict.
        Of these, 14 diagnostic classes are of primary interest, with values:
            Nan: no diagnostic marking available
            -1:  diagnosis uncertain
            0:   negative
            1:   positive
        We re-assign Nan to -2 by default.

        By default, we also convert all labels to numeric values:
            - all Nan are reassigned to -2
            - Frontal/Lateral:  Frontal = 1, Lateral = 2.
            - AP/PA:  nan=-2, AP=1, PA=2, LL=3, RL=4
            - Sex: Unknown= 1, Female=2, Male=3


    """
    LABELS_DIAGNOSTIC = [
        'No Finding',
        'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
        'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
        'Pleural Other', 'Fracture', 'Support Devices']
    LABELS_ALL = ['Sex', 'Age', 'Frontal/Lateral', 'AP/PA'] + LABELS_DIAGNOSTIC

    LABEL_CLEANUP_DICT = {col: {np.nan: -2} for col in LABELS_ALL}
    LABEL_CLEANUP_DICT['Frontal/Lateral'].update(
        {'Frontal': 1, 'Lateral': 2})
    LABEL_CLEANUP_DICT['AP/PA'].update(
        {'AP': 1, 'PA': 2, 'LL': 3, 'RL': 4})
    LABEL_CLEANUP_DICT['Sex'].update(
        {'Unknown': 1, 'Female': 2, 'Male': 3})

    @staticmethod
    def format_labels(getitem_dct: dict, labels=LABELS_ALL,
                      ret_type=T.tensor, label_cleanup_dct=LABEL_CLEANUP_DICT):
        """Helper method for converting the labels into a tensor or numpy array.

        :dct: the dict received in getitem_transform.
        :labels: either "all" or "diagnostic" or an ordered list of label names.
            ['Sex', 'Age', 'Frontal/Lateral', 'AP/PA', 'No Finding',
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung
            Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
            'Support Devices'].
        :ret_type: np.array or torch.tensor  (pytorch tensor by default)
        :returns: np.array or torch.tensor of numeric label values
        """
        lab = T.tensor(getitem_dct['labels'][labels])
        return lab

    def __init__(self, dataset_dir="./data/CheXpert-v1.0/",
                 use_train_set=True,
                 img_transform=tvt.ToTensor(),
                 getitem_transform=lambda x: (
                     x['image'], CheXpert.format_labels(x)),
                 label_cleanup_dct=LABEL_CLEANUP_DICT
                 ):

        train_or_valid = 'train' if use_train_set else 'valid'
        img_fp_glob = f"{dataset_dir.rstrip('/')}/{train_or_valid}/patient*/study*/*.jpg"
        label_fp = f"{dataset_dir.rstrip('/')}/{train_or_valid}.csv"
        self.labels_csv = pd.read_csv(label_fp).set_index('Path')

        super().__init__(img_fp_glob, img_transform)

        # join the labels_csv Path column to the actual filepath
        match = re.search(
            f'CheXpert-v1.0(-small)?/{train_or_valid}/patient', self.fps[0])
        if match is None:
            raise Exception((
                "The directory containing CheXpert data should"
                " have either of these names, spelled exactly like this:"
                " CheXpert-v1.0 or CheXpert-v1.0-small."
                " Please pass a correct `dataset_dir`"))
        self.idx_for_fp_to_csv_matching = match.start()
        self.__getitem_transform = getitem_transform

        # clean up the labels csv
        if label_cleanup_dct is not None:
            self.labels_csv.replace(label_cleanup_dct, inplace=True)


    def __getitem__(self, index, _getitem_transform=True):
        sample = super().__getitem__(index)
        sample['labels'] = self.labels_csv.loc[
            sample['fp'][self.idx_for_fp_to_csv_matching:]]
        if _getitem_transform and self.__getitem_transform is not None:
            return self.__getitem_transform(sample)
        else:
            return sample


class CheXpert_Small(CheXpert):
    def __init__(self, dataset_dir="./data/CheXpert-v1.0-small/", **kwargs):
        super().__init__(dataset_dir=dataset_dir, **kwargs)


if __name__ == "__main__":
    """
    to test this code, run
    $ ipython -im simplepytorch.datasets.chexpert
    """

    #  dset = CheXpert()
    dset = CheXpert_Small(use_train_set=False)
    print('z = dset[0] ; print(img, av_mask = z)')
    z = dset[0]
    print('image x-ray size:', z[0].shape)
    print('labels', z[1])
