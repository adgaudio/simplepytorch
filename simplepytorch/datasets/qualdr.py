import json
import os.path
try:
    import pyjq
except ModuleNotFoundError:
    import logging
    logging.warn("QualDR dataset labels unavailable because pyjq not installed")
import random
import statistics
import torchvision.transforms as tvt
import torch

from .glob_image_dir import GlobImageDir


class QualDR_Grading:
    """Collection of functions useful if you have a model that outputs
    a vector of all 13 GRADES predictions"""
    GRADES = {  # image level labels assigned by Ophthalmologists.
                # 0 is healthy. -1 is ungradable.
        'R0': 0, 'R1': 1, 'R2_M': 2, 'R2_S': 3, 'R3': 4, 'RX': -1,  # Diabetic Retinopathy
        'M0': 0, 'M1': 1, 'MX': -1,  # Macular Degeneration
        'P0': 0, 'P1': 1, 'P2': 2, 'PX': -1}  # Photocoagulation
    GRADES_FLAT = dict(zip(
        ['R0', 'R1', 'R2_M', 'R2_S', 'R3', 'RX', 'M0', 'M1', 'MX', 'P0', 'P1', 'P2',
        'PX'], range(13)))
    GRADES_FLAT_IDX_RANGES = [
        ("retinopathy", (0, 6)), ("maculopathy", (6, 9)), ("photocoagulation", (9, 13))]
    GRADE_CATEGORIES = ['retinopathy', 'maculopathy', 'photocoagulation']

    TRAIN_SET_CLASS_DISTRIBUTION = [  # classes correspond with GRADES_FLAT
        ('retinopathy', [0.3799759, 0.19059107, 0.22557297, 0.05066345, 0.0880579, 0.06513872]),
        ('maculopathy', [0.6031363, 0.1230398, 0.2738239]),
        ('photocoagulation', [0.8866104, 0.013269, 0.01206273, 0.0880579]),
    ]

    @staticmethod
    def create_confusion_matrices(y, yhat, hard_assignment=True):
        """Build one confusion matrix (CM) for each of the three grade categories,
        returning a dict with one confusion matrix per category.

        `hard_assignment` - By default, use a hard assignment where the
            prediction vector is converted to one-hot for each category.
            If False, use soft assignment where the prediction vector is
            converted to a probability distribution for each category.

        Assume the getitem_transform is the default.

        each row corresponds to a true class.
        and is a vector representing the distribution density of
        predicted values when that class was the correct one.
        """
        num_classes = QualDR_Grading.GRADES_FLAT_IDX_RANGES[-1][-1][-1]
        batch_size = y.shape[0]
        # --> get the index of the true class for each of the three grade types
        y_true_idxs = torch.arange(num_classes).repeat(batch_size, 1)[y==1].reshape(batch_size, 3)
        y_true_idxs -= torch.tensor([0] + [x[1][1] for x in QualDR_Grading.GRADES_FLAT_IDX_RANGES[:-1]])
        # --> obtain confusion matrices
        def get_CM(grade_name, grade_idx, class_start_idx, class_end_idx):
            """build a confusion matrix for grades:  retinopathy, maculopathy or photocoagulation
            """
            s, e = class_start_idx, class_end_idx
            yh = yhat[:, s:e]  # get the relevant predictions
            if hard_assignment:
                _onehot = torch.zeros_like(yh)
                _onehot[torch.arange(e-s).repeat(batch_size, 1) == yh.argmax(1).reshape(-1, 1)] = 1
                yh = _onehot
            yh = (yh - yh.min(1,True)[0])
            yh = yh / (yh.sum(1,True))  # convert to prob dist
            assert torch.allclose(yh.sum(), torch.tensor(batch_size).float()), 'bug'
            assert torch.allclose(yh.sum(1), torch.tensor(1).float()), 'bug'
            # add prediction vector to the appropriate true class vector
            CM = torch.zeros((e-s, e-s))
            for k in range(e-s):
                density = yh[y_true_idxs[:, grade_idx] == k].sum(0)
                CM[k, :] = density
            return CM
        CMs = {k: get_CM(k, i, s, e) for i, (k,(s,e)) in enumerate(QualDR_Grading.GRADES_FLAT_IDX_RANGES)}
        return CMs

    @staticmethod
    def loss_cross_entropy_per_category(input, target):
        target = target.long()
        losses = torch.stack([
            torch.nn.functional.cross_entropy(
                input[:, s:e], target[:, s:e].argmax(1),
                #  weight=(1-torch.tensor(weights, device=target.device))/(len(weights)-1))
                weight=max(weights)/torch.tensor(weights, device=target.device))
            #  )
            for ((k, (s,e)), (_, weights)) in zip(QualDR_Grading.GRADES_FLAT_IDX_RANGES,
                                                  QualDR_Grading.TRAIN_SET_CLASS_DISTRIBUTION)]
        ) / len(QualDR_Grading.GRADE_CATEGORIES)
        losses.backward = losses.sum().backward
        return losses

    @staticmethod
    def get_diagnosis_grades(
            json_data, diagnosis_grades=('retinopathy', 'maculopathy', 'photocoagulation')):
        lst = pyjq.all(
            f'''.annotations|select(.[].diagnosis != null)
            |map(select(.diagnosis!=null)
            |.diagnosis|with_entries(select([.key]|inside({json.dumps(diagnosis_grades)}))))[]''',
            json_data)

        if not lst:
            grades = ['RX', 'MX', 'PX']
        else:
            grades = [statistics.mode([x[k] for x in lst]) for k in diagnosis_grades]
        #  return torch.tensor([GRADES_FLAT[x] for x in grades], dtype=torch.long)
        rv = torch.zeros(13, dtype=torch.float)
        rv[[QualDR_Grading.GRADES_FLAT[g] for g in grades]] = 1
        return rv


class QualDR(GlobImageDir):
    """Load the ARSN QualDR dataset.  This dataset is not publicly available.

    A common usage looks like this:

        >>> dset = QualDR(
            img_transform=tvt.Compose([
                tvt.RandomCrop((512, 512)),
                tvt.ToTensor(),
            ]))
        )
    """
    def __init__(self, base_dir="./data/arsn_qualdr",
                 default_set='train',
                 img_transform=tvt.ToTensor(),
                 getitem_transform=lambda x: (
                    x['image'], QualDR_Grading.get_diagnosis_grades(x['json'])),
                 ):

        img_glob_expr = f"{base_dir}/imgs*/*png"
        self._image_loader = GlobImageDir(img_glob_expr, img_transform)

        self._json_fps = {
            img_fp: f"{base_dir}/annotations/{os.path.basename(img_fp)[:-4]}.json"
            for img_fp in self._image_loader.fps}

        self.__getitem_transform = getitem_transform

        # set up the train and test sets
        _idxs = list(range(len(self._image_loader.fps)))
        if default_set == 'both':
            self._train_or_test_index = _idxs
        else:
            assert default_set in {'train', 'test'}
            random.Random(48995).shuffle(_idxs) ; assert _idxs[0] == 961
            _train_idxs = _idxs[:int(.7*(len(_idxs)))]
            _test_idxs = _idxs[int(.7*(len(_idxs))):]
            assert len(_train_idxs) + len(_test_idxs) == len(self._image_loader.fps)
            self._train_or_test_index = _train_idxs if default_set == 'train' else _test_idxs

    def get_fps(self, idxs=None):
        """
        Return filepaths for either the test set or train set, depending on
        which is active.  Return a tuple (img_fps, annotation_fps)
        """
        if idxs is None:
            idxs = self._train_or_test_index
        img_fps = [self._image_loader.fps[x] for x in idxs]
        json_fps = [self._json_fps[k] for k in img_fps]
        return img_fps, json_fps

    def __getitem__(self, index, _getitem_transform=True):
        sample = self._image_loader[self._train_or_test_index[index]]
        sample['json'] = self._load_json(sample['fp'])
        if _getitem_transform and self.__getitem_transform is not None:
            return self.__getitem_transform(sample)
        else:
            return sample

    def _load_json(self, img_fp):
        with open(self._json_fps[img_fp], 'r') as fread:
            js = json.load(fread)
        return js

    def __len__(self):
        return len(self._train_or_test_index)


if __name__ == "__main__":
    # python -m simplepytorch.datasets.qualdr
    d = QualDR()

    z = d[0]
    print('d = QualDR() ; z = d[0] ; print(z[0].shape, z[1])')
    print('img', z[0].shape, 'grades', z[1])

    # generate a latex table containing the distribution of grades for each
    # class.
    #  import pandas as pd ;
    #  from simplepytorch.datasets import QualDR
    #
    #  dfs = {}
    #  for ts in [True, False]:
    #      d = QualDR(use_train_set=ts)
    #      grades = [x[1] for x in d]
    #      df = pd.DataFrame(grades, columns=['DR', 'MD', 'PC'])
    #      print('use_train_set =', ts)
    #      print(df.apply(lambda x: x.value_counts()).to_latex())
    #      dfs[ts and 'Train' or 'Test'] = df
    #  dff = pd.concat(dfs)
    #  counts = dff.unstack(0).apply(lambda x: x.value_counts())
    #  # format into a latex table
    #  rv = (counts.astype('str').applymap(lambda x: x.replace('.0', '').ljust(3)) + ('(' + (counts / counts.sum(0)*100).round(1).astype(str) + '%)').applymap(lambda x: x.ljust(1))).replace('nan(nan%)', '---')
    #  print('% version 1')
    #  print(rv.to_latex())
    #  print('% version 2')
    #  print(counts.astype('str').applymap(lambda x: x.replace('.0', '').replace('nan', '---')).to_latex())
