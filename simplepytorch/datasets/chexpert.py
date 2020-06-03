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
        We re-assign Nan to -2 by default for diagnostic classes.

        By default, we also convert all labels to numeric values:
            - all Nan for diagnostic classes are reassigned to -2
            - Frontal/Lateral:  Frontal = 0, Lateral = 1.
            - AP/PA:  nan=-1, AP=0, PA=1, LL=2, RL=3
            - Sex: Unknown=0, Female=1, Male=2


    """
    LABELS_DIAGNOSTIC = [
        'No Finding',
        'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
        'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
        'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
        'Pleural Other', 'Fracture', 'Support Devices']
    LABELS_METADATA = ['Sex', 'Age', 'Frontal/Lateral', 'AP/PA']
    LABELS_ALL = LABELS_METADATA + LABELS_DIAGNOSTIC

    LABEL_CLEANUP_DICT = {
        col: {0:0, 1:1, -1:2, np.nan: 3, } for col in LABELS_DIAGNOSTIC}
    LABEL_CLEANUP_DICT.update({
        'Frontal/Lateral': {'Frontal': 0, 'Lateral': 1}, 
        'AP/PA': {'AP': 0, 'PA': 1, 'LL': 2, 'RL': 3, np.nan: 4},
        'Sex': {'Female': 0, 'Male': 1, 'Unknown': 2},
        'Age': {i: i for i in range(90)}})

    @staticmethod
    def format_labels(getitem_dct: dict, labels=LABELS_ALL, explode=False):
        """Helper method for converting the labels into a tensor or numpy array.

        :dct: the dict received in getitem_transform.
        :labels: either "all" or "diagnostic" or an ordered list of label names.
            ['Sex', 'Age', 'Frontal/Lateral', 'AP/PA', 'No Finding',
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung
            Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
            'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
            'Support Devices'].
        :explode: If true, convert each class into a one-hot vector.
          For instance, 'Age' would get expanded into a one-hot vector of 90
          ages.  The diagnostic classes are expanded to four values
          corresponding to [0 (neg), 1 (pos), 2 (uncertain), 3 (blank)].

        :returns: If explode=False, return torch.tensor of numeric label values.
            If explode=True, return [(label_name: torch.tensor(one_hot)), ...]
        """
        if explode:
            y = []
            for lname in labels:
                tmp = T.zeros(
                    len(CheXpert.LABEL_CLEANUP_DICT[lname]), dtype=T.int8)
                tmp[getitem_dct['labels'][lname]] = 1
                y.append(tmp)
            y = T.cat(y)
        else:
            y = T.tensor(getitem_dct['labels'][labels])
        return y

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
            self.labels_csv = self.labels_csv.astype('int8')



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
    dset = CheXpert_Small(use_train_set=False, getitem_transform=lambda x: (x['image'], CheXpert.format_labels(x, explode=True)))
    print('z = dset[0] ; print(img, av_mask = z)')
    z = dset[0]
    print('image x-ray size:', z[0].shape)
    print('labels', z[1])
