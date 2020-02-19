import glob
import pandas as pd
import PIL
import re
import os.path
import torchvision.transforms as tvt
import torch

from simplepytorch.datasets._shared import as_tensor


class IDRiD_Segmentation:
    """
    Indian Diabetic Retinopathy Segmentation Dataset
    https://idrid.grand-challenge.org/

    A common usage looks like this:

        >>> idrid = IDRiD_Segmentation(
            getitem_transform=tvt.Compose([
                IDRiD_Segmentation.as_tensor(('MA', 'HE')),
                ... your transforms of the tensor of image and masks here
                ])
        )
        >>> sample = idrid[0]
        >>> img, MA, HE = sample[:3], sample[4], sample[5]

    getitem_transform formats the data into the form you want.
      If getitem_transform=None, it outputs a dictionary.  the `as_tensor` method stacks the image and masks together
    """
    POSITIVE_PIXELS_PER_CATEGORY_TRAIN = torch.tensor([
        704963.,  6500485.,  5319524.,  1249041., 11906334.])
    def __init__(self, dataset_dir="./data/IDRiD_segmentation",
                 use_train_set=True,
                 img_transform=None,
                 getitem_transform=as_tensor(
                     ('MA', 'HE', 'EX', 'SE', 'OD'),
                     include_image=True, return_numpy_array=False)):
        _labels = {
            'MA': '1. Microaneurysms',
            'HE': '2. Haemorrhages',
            'EX': '3. Hard Exudates',
            'SE': '4. Soft Exudates',
            'OD': '5. Optic Disc'}
        dset = r'a. Training Set' if use_train_set else r'b. Testing Set'

        # get the list of filepaths for each image.
        self.fps_img = {
            self._get_img_id(fp): fp
            for fp in sorted(glob.glob(f"{dataset_dir}/1. Original Images/{dset}/*jpg"))}
        self._fps_label = { # {MA: {IDRID_01: fp, ...}, ...}
            k: {self._get_img_id(fp): fp
            for fp in sorted(glob.glob(f"{dataset_dir}/2. All Segmentation Groundtruths/{dset}/{v}/*tif"))}
            for k, v in _labels.items()}
        self.fps_label = {
            img_id: {lesion_name: _fps[img_id] for lesion_name, _fps in self._fps_label.items() if img_id in _fps}
            for img_id in self.fps_img}
        self.idx_to_img_id = list(sorted(self.fps_img))

        self.img_transform = img_transform
        self.getitem_transform = getitem_transform

    def __getitem__(self, index, _getitem_transform=True):
        img_id = self.idx_to_img_id[index]
        img = self._load(self.fps_img[img_id])

        rv = {lesion_id: self._load(fp)
              for lesion_id, fp in self.fps_label[img_id].items()}
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
        return re.search('/(IDRiD_\d+)(_[A-Z]{2})?\.(tif|jpg)$', fp).group(1)

    LABELS = ('MA', 'HE', 'EX', 'SE', 'OD')
    @staticmethod
    def as_tensor(label_ids=LABELS,
                  include_image=True, return_numpy_array=False):
        """Meant to be used for getitem_transform to stack the image and
            segmentation masks into a tensor"""
        return as_tensor(label_ids, include_image, return_numpy_array)


if __name__ == "__main__":
    # ipython -im simplepytorch.datasets.idrid
    dset = IDRiD_Segmentation()

    print('dset = IDRiD_Segmentation() ; z = dset[0] ; print(z.shape)')
    z = dset[0]
    print(z.shape)

    img, MA, HE = z[:3], z[4], z[5]
