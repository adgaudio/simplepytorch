import torch


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
