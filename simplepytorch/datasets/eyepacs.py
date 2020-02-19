import pandas as pd
import os.path
import torchvision.transforms as tvt
from .glob_image_dir import GlobImageDir


class Eyepacs(GlobImageDir):
    """Load Eyepacs Dataset available for download from Kaggle, applying given
    transforms.  This loads the images under the "train" directory, for which
    we are given labels.

    A common usage looks like this:

        >>> eyepacs = Eyepacs(
            img_transform=tvt.Compose([
                tvt.RandomCrop((512, 512)),
                tvt.ToTensor(),
            ]),
            getitem_transform=lambda x: (
                x['image'],
                torch.tensor([int(x['Retinopathy grade'] != 0)]))
        )
    """
    def __init__(self, img_glob_expr="./data/eyepacs/train/*.jpeg",
                 label_csv_fp="./data/eyepacs/trainLabels.csv.zip",
                 img_transform=tvt.ToTensor(),
                 getitem_transform=lambda x: (
                    x['image'], x['Retinopathy grade'])):
        super().__init__(img_glob_expr, img_transform)
        self.__getitem_transform = getitem_transform

        self.csv_data = pd.read_csv(label_csv_fp)\
            .rename(columns={'image': 'Image ID',
                             'level': 'Retinopathy grade'})\
            .set_index('Image ID')

    def __getitem__(self, index, _getitem_transform=True):
        sample = super().__getitem__(index)
        image_id = os.path.basename(sample['fp'])[:-5]
        sample.update(dict(self.csv_data.loc[image_id]))
        sample['image_id'] = image_id
        if _getitem_transform and self.__getitem_transform is not None:
            return self.__getitem_transform(sample)
        else:
            return sample


class EyepacsTest(GlobImageDir):
    """Load Eyepacs Dataset available for download from Kaggle, applying given
    transforms.

    This loads the images under the "test" directory, for which
    we are not given labels.
    """
    def __init__(self, img_glob_expr="./data/eyepacs/test/*.jpeg",
                 img_transform=tvt.ToTensor(), getitem_transform=None):
        super().__init__(img_glob_expr, img_transform)


if __name__ == "__main__":
    # python -m simplepytorch.datasets.eyepacs
    d = Eyepacs()

    z = d[0]
    print(z['image'].shape)
