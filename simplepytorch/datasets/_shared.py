import numpy as np
import torch
import torchvision.transforms as tvt


def as_tensor(label_ids, include_image: bool, return_numpy_array: bool):
    """Helpful getitem transform for Segmantation datasets where the labels are
    same size as input image and we wish to stack the image and selected labels
    together into a single tensor.

    This function receives arguments in two steps.  First, receive a list dict
    keys whose values will be stacked into a tensor.  Second, receive a dict
    {'image': image_tensor, 'key1': ...}.  If the keys don't exist, fill with
    all zeros.

    The returned function will return a torch.tensor that stacks
    the image and given dct items identified by the label_ids

        my_tensor = as_tensor(('ground_truth_label1', 'label2'))({'image': img1, 'ground_truth_label1': img2, 'label2': img3})
    """
    def _to_numpy(img):
        if isinstance(img, torch.Tensor):
            assert len(img.shape) == 3, "to_tensor expects 3 channel image"
            return img.permute(1,2,0).numpy()
        elif not isinstance(img, np.ndarray):
            return np.array(img)
        return img
    def _as_tensor(dct):
        if include_image:
            lst = [_to_numpy(dct['image'])]
            shape = lst[0].shape[:2]
            dtype = lst[0].dtype
        else:
            lst = []
            tmp = _to_numpy(dct[[k for k in label_ids if k in dct][0]])
            shape = tmp.shape[:2]
            dtype = None  # assume don't know
        zeros = np.zeros(shape, dtype=dtype)[:,:,np.newaxis]
        lst.extend(_to_numpy(dct.get(k, zeros)) for k in label_ids)
        arr = np.dstack(lst)
        if return_numpy_array:
            return arr
        return tvt.functional.to_tensor(arr)
    return _as_tensor


