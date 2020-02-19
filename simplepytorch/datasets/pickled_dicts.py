import torch
import glob
import os.path as P
import pickle


class PickledDicts(torch.utils.data.Dataset):
    """Load a directory of pickled python dictionaries.
    Each dict must have at least an "image" key.
    """

    def __init__(self, base_dir, img_transform, getitem_transform):
        self.base_dir = base_dir
        self.img_transform = img_transform
        self.getitem_transform = getitem_transform
        self.fps = glob.glob(P.join(base_dir, '*'))

    def __len__(self):
        return len(self.fps)

    def _load_fp(self, fp):
        with open(fp, 'rb') as fin:
            return pickle.load(fin)

    def __getitem__(self, index):
        dct = self._load_fp(self.fps[index])
        if self.img_transform is not None:
            dct['image'] = self.img_transform(dct['image'])
        if self.getitem_transform:
            dct = self.getitem_transform(dct)
        return dct


if __name__ == "__main__":
    # view images from a pickled dicts dataset
    import os
    import sys
    import random
    import pickle
    from matplotlib import pyplot as plt


    dirp = sys.argv[1]
    fns = os.listdir(dirp)
    random.shuffle(fns)


    for fname in fns:
        with open(P.join(dirp, fname), 'rb') as fin:
            dct = pickle.load(fin)
        print(dct['image'].max(), dct['image'].min())
        plt.imshow(dct['image'])
        plt.pause(0.5)
