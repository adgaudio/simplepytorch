import torch as T
import pandas as pd
from typing import Tuple
import PIL
import os.path
import numpy as np
import glob


class BBBC038v1(T.utils.data.Dataset):
    """
    BBBC038v1, also known as Kaggle Data Science Bowl 2018, is made available
    by the Broad Bioimage Benchmark Collection.

    It is a microscopy dataset enabling nuclei instance segmentation.

    Broad Bioimage Benchmark Collection [Caicedo et al., Nature Methods, 2019]."

    Expects a directory structure like:
        ./data/BBBC038v1_microscopy/
            ./stage1_train/
                ... only the unzipped contents of stage1_train.zip here.
            ./stage1_test/
                ... only the unzipped contents of stage1_test.zip here.
            ./stage2_test_final/
                ... only the unzipped contents of stage2_test_final.zip here.

    """
    def __init__(self, partition='stage1_train',
                 base_dir=os.path.abspath(os.environ.get("BBBC038v1_base_dir", './data/BBBC038v1_microscopy')),
                 convert_to=None):
        """
        :partition:  a value in ['stage1_train', 'stage1_test', 'stage2_test_final']
        :base_dir: location of dataset on disk.
        :convert_to:  a value in [T.Tensor, np.ndarray, None] that determines
            the output type of the images and masks.
        """
        self.img_ids = os.listdir(f'{base_dir}/{partition}/')
        self.fps = [
            f'{base_dir}/{partition}/{img_id}/images/{img_id}.png'
            for img_id in self.img_ids]
        self._convert_to = convert_to
        self.partition = partition  # for convenience

        if partition == 'stage1_train':
            df = pd.read_csv(f'{base_dir}/stage1_train_labels.csv')
        elif partition == 'stage1_test':
            df = pd.read_csv(f'{base_dir}/stage1_solution.csv')
        else:
            df = None
        self.labels_csv = df

    def __repr__(self):
        return f'{self.__class__.__name__}({self.partition}, ...)<{len(self)} images>'

    def __len__(self):
        return len(self.fps)

    def get(self, img_id):
        return self[self.img_ids.index(img_id)]

    def reconstruct_masks(self, shape, img_id):
        for rle_str in self.labels_csv.query(f'ImageId == "{img_id}"')['EncodedPixels']:
            yield self.rle_decode(shape, rle_str)

    def __getitem__(self, index):
        img_fp = self.fps[index]
        with PIL.Image.open(img_fp, 'r') as img:
            img.load()
        masks = list(self.reconstruct_masks(np.array(img).shape[:2], self.img_ids[index]))
        if not self._convert_to:
            masks = [PIL.Image.fromarray(x) for x in masks]
        else:
            img, masks = np.array(img)[:,:,:3], np.dstack(masks)
            #  assert (masks.round() == masks).all()
            if self._convert_to in ('tensor', T.Tensor):
                # assume NO label noise in masks.  Expect binary values!
                img, masks = T.tensor(img).permute(2,0,1), T.tensor(masks, dtype=T.long).permute(2,0,1)
        return img, masks

    @staticmethod
    def rle_encode(mask: np.ndarray) -> str:
        """
        Generate the run length encoding described in evaluation section of the
        kaggle website.  It's one-indexed top to bottom, left to right.

        Also described at https://www.kaggle.com/stainsby/fast-tested-rle-and-input-routines

        :mask: a binary mask like one that specifies if y == yhat
        """
        rle = np.diff(np.pad(mask.T.reshape(-1), (1,1))).nonzero()[0] + 1
        rle[1::2] -= rle[0::2]
        return ' '.join(rle.astype(str))

    @staticmethod
    def rle_decode(shape: Tuple[int], rle_string: str) -> np.ndarray:
        """Convert the run length encoding into a binary segmentation mask,
        as defined in the kaggle competition, so that we can generate
        segmentation masks for the test set."""
        rle = np.fromstring(rle_string, sep=' ', dtype=int)
        rle = rle.reshape(-1,2)
        rle[:,1] += rle[:,0]  # change from offset to actual index
        rle -= 1  # make 0 indexed
        x = np.zeros(np.prod(shape), dtype=bool)
        for start,end in rle:
            x[start:end] = True
        x = x.reshape(shape[1], shape[0]).T
        return x


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    dset_train = BBBC038v1('stage1_train', convert_to='numpy')
    dset_val = BBBC038v1('stage1_test', convert_to='tensor')
    dset_eval_on_kaggle = BBBC038v1('stage2_test_final', convert_to=None)

    x,ys = dset_train[0]

    # show generated labels are correctly aligned (tests the decoder)
    y = (ys * np.arange(1,1+ys.shape[0]).reshape(-1,1,1)).sum(0)
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(x)
    axs[1].imshow(y)

    # try to encode and then decode (tests the encoder)
    x,ys = dset_train.get('00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552')
    print('RLE Encoding', BBBC038v1.rle_encode(ys[0]))
    a = BBBC038v1.rle_decode(x.shape[:2], BBBC038v1.rle_encode(ys[1]))
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(a*1., 'gray')
    axs[1].imshow(ys[1]*100+x.mean(-1), 'gray')#.reshape(*a.shape, 1)*10. +x)
    axs[2].imshow(x)
    plt.show()
