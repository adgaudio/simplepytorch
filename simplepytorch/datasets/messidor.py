import pandas as pd
import os.path
import glob
import numpy as np
import torchvision.transforms as tvt
from .glob_image_dir import GlobImageDir
from .train_val_test_split import train_val_test_split


class Messidor(GlobImageDir):
    """
    Messidor Dataset available for download from
    http://www.adcis.net/en/third-party/messidor/

    Load Messidor Dataset, applying given transforms.
    Assume data is at ./data/messidor/*.csv and ./data/messidor/**/*.tif

    getitem_transform - Choose what to return (ie label and image).  If None, return a dict.
    img_transform - How to transform images

    A common usage looks like this:

        >>> messidor = Messidor(
            img_transform=tvt.Compose([
                tvt.RandomCrop((512, 512)),
                tvt.ToTensor(),
            ]),
            getitem_transform=lambda x: (
                x['image'],
                torch.tensor([int(x['Retinopathy grade'] != 0)]))
        )
    """
    def __init__(self, csv_glob_expr='./data/messidor/*.csv',
                 img_glob_expr='./data/messidor/**/*.tif',
                 img_transform=tvt.ToTensor(),
                 getitem_transform=None):
        super().__init__(img_glob_expr, img_transform)
        self.__getitem_transform = getitem_transform
        self.csv_data = pd.concat([
            pd.read_csv(x) for x in glob.glob(csv_glob_expr, recursive=True)])\
            .set_index('Image name')
        assert self.csv_data.shape[0] == len(self.fps)  # sanity check
        self.shape_data = None  # populate this requires pass through all imgs

    def __getitem__(self, index, _getitem_transform=True):
        sample = super().__getitem__(index)
        fname = os.path.basename(sample['fp'])
        sample.update(dict(self.csv_data.loc[fname]))
        if _getitem_transform and self.__getitem_transform is not None:
            return self.__getitem_transform(sample)
        else:
            return sample

    def getitem_no_transform(self, index):
        """Apply the image transform, but not the getitem_transform.
        Return a dict
        """
        return self.__getitem__(index, False)

    def train_test_split(
            self, train_frac, test_frac=0.2, random_state_test=27204):
        """
        Train val test split and STRATIFY across the Opthalmologic departments
        that the images came from because the dimensions of images from each
        department are different.
        If a validation set is to be created, train_frac + test_frac < 1.
        By default, test set is a persistent 20% of data.

        train_frac, test_frac: a value in [0, 1]
        random_state: can be used to create a test set that is persistent
            across processes and machines.  Enabled by default
        """
        arr = np.arange(len(self))
        stratify = self.csv_data['Ophthalmologic department'].values
        return train_val_test_split(
            arr, train_frac=train_frac, test_frac=test_frac, stratify=stratify,
            random_state_test=random_state_test
        )

    def fetch_img_dims(self):
        """
        Iteratively load all images in dataset and store their shape
        in a dataframe.  Useful for analysis.  Takes a minute or so.

        #  # file dimensions are not uniform.
        #  # base 1 and base 2 have unique dimension.
        #  # base 3 has 2 different dimensions.
        #  df.groupby(['base', 'x', 'y', 'z'])['fp'].count()
        """
        df = pd.DataFrame(
            {fp: list(self[i].shape)
             for i, fp in zip(range(len(self.fps)), self.fps)})
        df.columns = ['fp', 'x', 'y', 'z']
        df = pd.concat([df, df['fp'].str.extract(
            r'/Base(?P<base>\d)(?P<base2>\d)/').astype('int')], axis=1)
        df['Image name'] = df['fp'].apply(os.path.basename)
        df.set_index('Image name')
        return df


if __name__ == "__main__":
    # python -m simplepytorch.datasets.messidor
    messidor = Messidor()
    z = messidor[0]
    print(np.array(z['image']).shape)
