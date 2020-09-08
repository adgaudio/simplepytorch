import glob
import PIL
import numpy as np
import re
import torchvision.transforms as tvt
import torch

from simplepytorch.datasets._shared import as_tensor


def crop_background(img_and_labels_and_mask: np.ndarray):
    mask = img_and_labels_and_mask[..., -1:]

    # crop img and labels using the mask.
    _tmp1 = (mask.sum(1)>0)
    _tmp2 = (mask.sum(0)>0)
    crop_slice = np.s_[
        _tmp1.argmax():_tmp1[::-1].argmax()*-1,
        _tmp2.argmax():_tmp2[::-1].argmax()*-1, :]
    return img_and_labels_and_mask[crop_slice]


def to_tensors(device):
    """
    Input a tensor of shape (h,w,3+n)
    Output a tuple of tensors ( (h,w,3), (h,w,1),  ..., (h,w,1) )
    """
    def _to_tensors(img_labels_mask: np.ndarray):
        img_labels_mask = img_labels_mask.transpose(2,0,1)

        img = torch.tensor(img_labels_mask[:3], dtype=torch.float, device=device)
        rv = [img]
        for i in range(3, img_labels_mask.shape[0]):
            rv.append(torch.tensor(img_labels_mask[i], dtype=torch.uint8, device=device))
        return rv
    return _to_tensors


class DRIVE:
    """
    DRIVE (...)
    ... URL

    A common usage looks like this:

        >>> dset = DRIVE(
            getitem_transform=tvt.Compose([
                DRIVE.as_tensor(['1st_manual', 'mask'], return_numpy_array=True),
                DRIVE.crop_background,
                DRIVE.to_tensors('cuda:0'),
                ])
        )
        >>> img, labels_1st_manual, mask = drive[0]

    getitem_transform formats the data into the form you want.
      If getitem_transform=None, it outputs a dictionary.

  NOTE: Labels from the '1st_manual' annotator are available for both training
  and test sets.  However, '2nd_manual' labels are available only for test set.
    """
    LABELS = ['1st_manual', '2nd_manual', 'mask']

    def __init__(self, dataset_dir="./data/DRIVE",
                 use_train_set=True,
                 img_transform=None,
                 getitem_transform=tvt.Compose([
                     as_tensor(['1st_manual', 'mask'], include_image=True, return_numpy_array=True),
                     crop_background,
                     to_tensors('cpu')
                 ])):
        dset = 'training' if use_train_set else 'test'
        _labels = ['1st_manual', 'mask'] if use_train_set else ['1st_manual', '2nd_manual', 'mask']

        # get the list of filepaths for each image.
        self.fps_img = {
            self._get_img_id(fp): fp
            for fp in sorted(glob.glob(f"{dataset_dir}/{dset}/images/*.tif"))}
        self._fps_label = {  # {'1st_manual': {img_id: fp}, '2nd_manual': ...}
            k: {self._get_img_id(fp): fp
                for fp in sorted(glob.glob(f"{dataset_dir}/{dset}/{k}/*gif"))}
            for k in _labels}
        self.fps_label = {
            img_id: {label_type: _fps[img_id] for label_type, _fps in self._fps_label.items()
                     if img_id in _fps}
            for img_id in self.fps_img}
        self.idx_to_img_id = list(sorted(self.fps_img))

        self.img_transform = img_transform
        self.getitem_transform = getitem_transform

    def __getitem__(self, index, _getitem_transform=True):
        img_id = self.idx_to_img_id[index]
        img = self._load(self.fps_img[img_id])

        rv = {label_type: self._load(fp)  # '1st_manual' or '2nd_manual' maps
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
        return re.search(r'(\d+)_((training|test)(.tif|_mask.gif)|manual[12].gif)$', fp).group(1)

    @staticmethod
    def as_tensor(label_ids=('1st_manual',),
                  include_image=True, return_numpy_array=False):
        """Meant to be used for getitem_transform to stack the image and
            segmentation masks into a tensor"""
        return as_tensor(label_ids, include_image, return_numpy_array)

    @staticmethod
    def crop_background(img_and_labels_and_mask: np.ndarray):
        """For use in getitem_transform.  Minimizes amount of black background.
        Removes the mask channel, which we assume is last channel of input array.

        Must be called after `DRIVE.as_tensor([..., 'mask'], return_numpy_array=True)`.
        For instance:

            getitem_transform=tvt.Compose([
                DRIVE.as_tensor(['1st_manual', 'mask'], return_numpy_array=True),
                DRIVE.crop_background,
                ... your preprocessing here
                DRIVE.to_tensors(device='cpu'), ])
        """
        return crop_background(img_and_labels_and_mask)

    @staticmethod
    def to_tensors(device):
        """For use in getitem_transform, after as_tensor(...,
        return_numpy_array).  Converts a numpy array to tensor and puts
        channels in first dimension.

        Returns (img, labels_1st_manual, [labels_2nd_manual], mask)
        """
        return to_tensors(device)


if __name__ == "__main__":
    # ipython -im simplepytorch.datasets.drive
    dset = DRIVE(use_train_set=True)

    img, labels_1st_manual, mask = dset[0]
    print(img.shape, labels_1st_manual.shape, mask.shape)
