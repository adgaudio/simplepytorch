import torch


def create_confusion_matrix(y, yhat, normalize_y=True):
    """Confusion Matrix enables performance evaluation of a classifier model.

    Each row corresponds to a true class and represents the
    distribution of predicted values when that class was the correct one.

    Let `n` represent the number of (minibatch) samples and `m` represent the
    number of classes.  Basically, compute the outer product `y y_{hat}^T` for
    each of the `n` samples and sum the resulting matrices.

    This is a bit more general than the typical description of a confusion
    matrix, in that the ground truth `y` values can be fractions, multi-label
    and not necessarily sum to 1.  The trade-off is that y and yhat are never
    1-dimensional (e.g. for binary classification, both y and yhat have two
    columns).

    :y:  (n,m) matrix of true values (e.g. if multi-class, each row is one-hot).
        In general, it doesn't make sense if y has negative values.
        e.g. for binary classification, set m=2.
    :yhat:  (n,m) matrix of predicted values.  Note: All `m` classes should
        have the same bounds (ie they are all logits).
        e.g. `yhat = softmax(model(x))`
        or a multi-class setting: `yhat = model(x).argmax(1)`
    :normalize_y:  If True, each row of `y` is converted to a probability
        distribution to determine what fraction of `yhat` to assign to each
        row.  If False, y is not normalized.  False is useful if `y` encodes
        the notion that different samples have different weights; the
        attribution of yhat vector to the relevant classes for sample A should
        have more contribution than those for sample B.  (Assumes the sum
        of a row of y never equals zero.)

    :returns: an (m,m) confusion matrix.
    """
    # --> convert each row of y into a probability distribution
    if normalize_y:
        w = y / y.sum(1, keepdims=True)
    else:
        w = y
    # --> compute outer product w yhat^T for each of the n samples.  This gives
    # a confusion matrix for each sample.  Sum the matrices.
    return torch.einsum('nm,no->mo', w, yhat)


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
