import PIL.Image
import glob
import torch.utils.data as TD


class GlobImageDir(TD.Dataset):
    """Load a dataset of files using a glob expression and Python Pillow
    library (PIL), and run optional transform func

    >>> GlobDir("./data/**/*.png")  # fetch PNG images recursively under ./data
    >>> GlobDir("./data/*/*.png")  # fetch images from the grandchild dirs
    >>> GlobDir("*.png", mytranform_fn)  # fetch and transform PNG files
    """

    def __init__(self, glob_expr, img_transform=None, getitem_transform=None):
        self.fps = glob.glob(glob_expr, recursive=True)
        self.__img_transform = img_transform
        self.__getitem_transform=getitem_transform

    def __len__(self):
        return len(self.fps)

    def __getitem__(self, index):
        fp = self.fps[index]
        with PIL.Image.open(fp) as im:
            im.load()
        if self.__img_transform:
            im = self.__img_transform(im)
        dct = {'image': im, 'fp': fp}
        if self.__getitem_transform is not None:
            return self.__getitem_transform(dct)
        else:
            return dct
