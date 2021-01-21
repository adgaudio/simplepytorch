import torch
from typing import Union


def confusion_matrix_1D_input(y: torch.LongTensor, yhat: torch.LongTensor, num_classes=None) -> torch.Tensor:
    assert not isinstance(y, (torch.FloatTensor, torch.cuda.FloatTensor)), 'y or yhat cannot have float values'
    assert not isinstance(yhat, (torch.FloatTensor, torch.cuda.FloatTensor)), 'y or yhat cannot have float values'
    if num_classes is None:
        num_classes = y.max()
    return torch.sparse_coo_tensor(
        torch.stack([y, yhat]),
        torch.ones(yhat.numel(), device=y.device),
        size=(num_classes, num_classes)).to_dense()

def confusion_matrix_binary_soft_assignment(
        y: Union[torch.FloatTensor,torch.LongTensor],
        yhat: Union[torch.FloatTensor,torch.LongTensor]):
    """
    Create a binary confusion matrix give 1d inputs, where the inputs are
    assumed to be probabilities of the (positive) class with index value 1.

    Performs a soft assignment of the confusion matrix.  It is useful for
    binary classification with label noise, or less commonly when the predicted
    probabilities are considered as soft assignment.  Here's an example input
    and output:

        >>> y = [.7]
        >>> yhat = [.2]
        >>> confusion_matrix(y, yhat, 2)
            [[(1-.7) * (1-.2) , (1-.7) * .2]
             [(.7    * (1-.2) , .7     * .2]]

    """

    # special case of 1D probability vectors
    assert (y.max() <= 1).all() and (y.min() >= 0).all()
    assert (yhat.max() <= 1).all() and (yhat.min() >= 0).all()

    yhat = torch.stack([1-yhat, yhat]).T.float()
    y = torch.stack([1-y, y]).T.float()
    return confusion_matrix_2D_input(y=y, yhat=yhat)


def confusion_matrix_2D_input(y: torch.Tensor, yhat: torch.Tensor, normalize_y=False) -> torch.Tensor:
    """
    Output a confusion matrix from 2D inputs.

    Inputs are assumed either both LongTensor or both FloatTensor.  In the
    LongTensor setting, the columns of input arrays y and yhat are class index
    and the rows correspond to different (minibatch) samples.  In FloatTensor
    setting, read below to understand better what you're doing.

    How: Compute the outer product `y y_{hat}^T` for
    each of the `n` samples and sum the resulting matrices.  In other words,
    each row is a weighted sum of the yhat row vector with the scalar weight
    y_i, so we have a row of the confusion matrix defined as (yhat * y_i).

    This function generalizes a confusion matrix, because the ground truth `y`
    values are free to be fractions, multi-label and do not need to sum to 1.
    The trade-off of this generalized design is that y and yhat are never
    1-D vectors, like they are in the sklearn method (e.g. for binary
    classification, both y and yhat have two columns).

    :y:  (n,c) matrix of true values (e.g. if multi-class, each row is one-hot).
        In general, it doesn't make sense if y has negative values.
        e.g. for binary classification, set c=2.
    :yhat:  (n,c) matrix of predicted values.  Note: All `c` classes should
        have the same bounds (ie they are all logits).
        e.g. `yhat = softmax(model(x))`
        or a multi-class setting: `yhat = model(x).argmax(1)`
    :normalize_y:  If True, ensure the ground truth labels (each row of `y`) is
        a probability distribution (sums to 1).  If False, y is not normalized.
        False is useful if different samples have different weights (Assumes the
        sum of a row of y never equals zero.)

    :returns: an (c,c) confusion matrix.
    """
    # --> convert each row of y into a probability distribution
    if normalize_y:
        w = y / y.sum(1, keepdims=True)
    else:
        w = y
    # --> compute outer product w yhat^T for each of the n samples.  This gives
    # a confusion matrix for each sample.  Sum the matrices.
    return torch.einsum('nm,no->mo', w, yhat)


def confusion_matrix(y: torch.Tensor, yhat: torch.Tensor, num_classes:int
                     ) -> torch.Tensor:
    """
    A Confusion Matrix enables performance evaluation of a classifier model's
    predictions.  This function works with multi-class or multi-label data, 1D
    input vectors or 2-D inputs.  Be very careful to give inputs of correct type.

    Each row index corresponds to a ground truth class.
    Each column index corresponds to predicted class.

    Standard example: Binary classification, where class "1" is positive, the
    output confusion matrix is:

        | TN | FP |
        | FN | TP |

        where "T" and "F" means true or false, "P" and "N" mean predicted
        positive or predicted negative

    Regarding the inputs:
        :y: ground truth tensor, shape is 1D or 2D.
        :yhat: prediction tensor, shape is 1D or 2D.

        If 1-D input tensor:
            - type must be either
              - LongTensor containing class index (i.e. This is the default
                supported case in sklearn for multi-class setting)
              - FloatTensor with probability of positive class is also allowed,
                but only when num_classes=2.  See "Special Case" section below.

            - The 1-D inputs force multi-class (one-hot) semantics, and is more
            or less the canonical setting.
            - Each value of the 1D vectors identifies a class in the
            confusion matrix by its row index or column index, for `y` and
            `yhat` respectively.

            ** Special Case:  Also support 1D floats when num_classes=2
              - In this case, assume that values are probabilities of
              (positive) class that is identified in the confusion matrix with
              index 1.  Performs a soft assignment of the confusion matrix.  It
              is useful for binary classification with label noise, or less
              commonly when the predicted probabilities are considered as
              soft assignment.  Here's an example input and output:

                >>> y = [.7]
                >>> yhat = [.2]
                >>> confusion_matrix(y, yhat, 2)
                    [[(1-.7) * (1-.2) , (1-.7) * .2]
                     [(.7    * (1-.2) , .7     * .2]]

        If 2-D input tensor:
            - Must be a LongTensor
            - Passing 2-D (n,c) inputs is better suited for multi-label
            data or uncertain ground truth labels or weights over samples.
              - `n` is the number of (minibatch) samples
              - `c` is the number of classes
            - If only one input is 2-D, the other is converted to 2-D.

        - Try to gracefully handle some funky shapes, such as (n,c,1,1) or (n,1)

        :returns: (c,c) tensor, where c is the number of classes.
    """
    # --> convert to 1d input if possible.  e.g. if shape is (n,1,1,1...)
    yhat = yhat.reshape(*yhat.shape[:2])
    y = y.reshape(*y.shape[:2])
    try: y = y.squeeze(1)
    except IndexError: pass
    try: yhat = yhat.squeeze(1)
    except IndexError: pass

    if 1 == yhat.ndim == y.ndim:
        if any(isinstance(x, (torch.FloatTensor, torch.cuda.FloatTensor))
               for x in [y, yhat]):
            assert num_classes == 2
            # special case for 1D probability vectors
            return confusion_matrix_binary_soft_assignment(y=y, yhat=yhat)
        else:
            # dispatch the function with 1D integer inputs
            return confusion_matrix_1D_input(y, yhat, num_classes)
    else:
        # dispatch the 2-D inputs function
        # --> but first, promote a 1D input to 2D if it exists
        if y.ndim == 1:
            y = _confusion_matrix_convert_2d_shape(y, num_classes)
        elif yhat.ndim == 1:
            yhat = _confusion_matrix_convert_2d_shape(yhat, num_classes)
        ret = confusion_matrix_2D_input(y, yhat)
        assert ret.shape == (num_classes, num_classes), "sanity check"
        return ret


def _confusion_matrix_convert_2d_shape(arr, num_classes):
    assert not isinstance(arr, (torch.FloatTensor, torch.cuda.FloatTensor))
    arr = torch.eye(num_classes, dtype=arr.dtype, device=arr.device)[arr]
    assert arr.shape[1] == num_classes, 'sanity check'
    assert arr.ndim == 2, 'sanity check'
    return arr


def accuracy(cm):
    return cm.trace() / cm.sum()


def matthews_correlation_coeff(cm):
    """
    Implementation of the R_k coefficient applied to a confusion matrix is
    matthew's correlation coefficient.
    Original implementation here:  http://rk.kvl.dk/software/rkorrC
    Section 2.3 of the Gorodkin paper.  https://www.ncbi.nlm.nih.gov/pubmed/15556477?dopt=Abstract
    """
    cm = cm.float()
    N = cm.sum()
    rowcol_sumprod = (cm@cm).sum()
    rowrow_sumprod = (cm@cm.T).sum()
    colcol_sumprod = (cm.T@cm).sum()
    cov_xy = N*cm.trace() - rowcol_sumprod
    cov_xx = N**2 - rowrow_sumprod
    cov_yy = N**2 - colcol_sumprod
    Rk = cov_xy / torch.sqrt(cov_xx)/torch.sqrt(cov_yy)
    return Rk

def precision(cm):
    rv = cm.diag() / cm.sum(0)
    rv[torch.isnan(rv)]=0
    return rv


def recall(cm):
    rv = cm.diag() / cm.sum(1)
    rv[torch.isnan(rv)]=0
    return rv
