import glob
import pandas as pd
import PIL
import re
import os.path
import torchvision.transforms as tvt
import torch

from simplepytorch.datasets._shared import as_tensor


class RITE:
    """
    RITE (Retinal Images vessel Tree Extraction)
    https://medicine.uiowa.edu/eye/rite-dataset

    A common usage looks like this:

        >>> rite = RITE(
            getitem_transform=tvt.Compose([
                RITE.as_tensor(['av', 'vessel']),
                ... your transforms of the tensor of image and masks here
                ])
        )
        >>> sample = rite[0]
        >>> img, av, vessles = sample[:3], sample[3], sample[4]

    getitem_transform formats the data into the form you want.
      If getitem_transform=None, it outputs a dictionary.
    """
    LABELS_AV = ['arteries', 'overlap', 'veins']
    def __init__(self, dataset_dir="./data/RITE",
                 use_train_set=True,
                 img_transform=None,
                 getitem_transform=as_tensor(
                     ['av'], include_image=True, return_numpy_array=False)):
        _labels = ['av', 'vessel']
        dset = 'training' if use_train_set else 'test'

        # get the list of filepaths for each image.
        self.fps_img = {
            self._get_img_id(fp): fp
            for fp in sorted(glob.glob(f"{dataset_dir}/{dset}/images/*.tif"))}
        self._fps_label = {  # {'vessel': {img_id: fp}, 'av': ...}
            k: {self._get_img_id(fp): fp
            for fp in sorted(glob.glob(f"{dataset_dir}/{dset}/{k}/*png"))}
            for k in _labels}
        self.fps_label = {
            img_id: {label_type: _fps[img_id] for label_type, _fps in self._fps_label.items() if img_id in _fps}
            for img_id in self.fps_img}
        self.idx_to_img_id = list(sorted(self.fps_img))

        self.img_transform = img_transform
        self.getitem_transform = getitem_transform

    def __getitem__(self, index, _getitem_transform=True):
        img_id = self.idx_to_img_id[index]
        img = self._load(self.fps_img[img_id])

        rv = {label_type: self._load(fp)  # 'av' or 'vessel' maps
              for label_type, fp in self.fps_label[img_id].items()}
        rv.update({
            'image': self.img_transform(img) if self.img_transform else img,
            'img_id': img_id})

        if _getitem_transform and self.getitem_transform is not None:
            return self.getitem_transform(rv)
        else:
            return rv

    def __len__(self):
        return len(self.fps_img)

    def _load(self, fp):
        with PIL.Image.open(fp) as im:
            im.load()
        return im

    def _get_img_id(self, fp):
        return re.search('(\d+)_(training|test).(tif|png)$', fp).group(1)

    @staticmethod
    def as_tensor(label_ids=('av',),
                  include_image=True, return_numpy_array=False):
        """Meant to be used for getitem_transform to stack the image and
            segmentation masks into a tensor"""
        return as_tensor(label_ids, include_image, return_numpy_array)


if __name__ == "__main__":
    # ipython -im simplepytorch.datasets.rite
    dset = RITE()

    print('z = dset[0] ; print(img, av_mask = z)')
    z = dset[0]
    img = z[:3]
    av_mask = z[3]
    print(img.shape, av_mask.shape)
