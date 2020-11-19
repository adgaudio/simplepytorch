import torch


def confusion_matrix_1D_input(y: torch.tensor, yhat: torch.Tensor, num_classes=None) -> torch.Tensor:
    if num_classes is None:
        num_classes = y.max()
    return torch.sparse_coo_tensor(
        torch.stack([y.round().long(), yhat.round().long()]),
        torch.ones(yhat.numel(), device=y.device),
        size=(num_classes, num_classes)).to_dense()


def confusion_matrix_2D_input(y: torch.Tensor, yhat: torch.Tensor, normalize_y=False) -> torch.Tensor:
    """
    Output a confusion matrix.

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


def confusion_matrix(y: torch.Tensor, yhat: torch.Tensor, num_classes:int,
                     normalize_y: bool = False,) -> torch.Tensor:
    """
    A Confusion Matrix enables performance evaluation of a classifier model's
    predictions.  This function works with multi-class or multi-label data, 1D
    input vectors or 2-D inputs.

    Each row index corresponds to a ground truth class.
    Each column index corresponds to predicted class.

    Standard example: Binary classification, where class "1" is positive, the
    output confusion matrix is:

        | TN | FP |
        | FN | TP |

        where "T" and "F" means true or false, "P" and "N" mean predicted
        positive or predicted negative

    Regarding the inputs:
        :y: ground truth tensor
        :yhat: prediction tensor

        If 1-D input tensor:
            :num_classes: Identifies how many classes in the confusion matrix,
            useful when `y` does not include all classes to guarantee output shape.

            - Each scalar value of the 1D vectors identifies a class in the
            confusion matrix by its row index or column index, for `y` and
            `yhat` respectively.
            - The 1-D inputs force multi-class (one-hot) semantics, and is more
            or less the canonical setting.

        If 2-D input tensor:

            Passing 2-D (n,c) inputs is better suited for multi-label
            data or uncertain ground truth labels or weights over samples.
              - `n` is the number of (minibatch) samples
              - `c` is the number of classes
            - All 2-D inputs are converted to float.
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
        # dispatch the function with 1D inputs
        return confusion_matrix_1D_input(y, yhat, num_classes)
    else:
        # dispatch the 2-D inputs function
        # --> but first, promote a 1D input to 2D if it exists
        if y.ndim == 1:
            y = _confusion_matrix_convert_2d_shape(y, num_classes)
        elif yhat.ndim == 1:
            yhat = _confusion_matrix_convert_2d_shape(yhat, num_classes)
        ret = confusion_matrix_2D_input(y, yhat, normalize_y)
        assert ret.shape == (num_classes, num_classes), "sanity check"
        return ret


def _confusion_matrix_convert_2d_shape(arr, num_classes):
    arr = torch.eye(num_classes, dtype=arr.dtype, device=arr.device)[arr.long()]
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
