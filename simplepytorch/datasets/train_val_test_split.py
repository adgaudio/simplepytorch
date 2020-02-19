from sklearn.model_selection import train_test_split


def train_val_test_split(
        idxs, train_frac, test_frac=None,
        random_state_test=None, stratify=None):
    """
    Split given idxs into train/val/test splits.

    If a validation set is to be created, train_frac + test_frac < 1.

    stratify: an array of categories to stratify the splits over
    random_state_test: seed or np.RandomState to ensure test set is persistent.
    """
    if test_frac is None:
        test_frac = 1 - train_frac
        assert 0 <= train_frac <= 1
    if train_frac + test_frac > 1:
        raise Exception(
            "train val test split: set train_frac + test_frac <= 1 please")
    if test_frac is None:
        test_frac = 1 - train_frac
    _trainval_idxs, test_idxs = train_test_split(
        idxs, train_size=1-test_frac, test_size=test_frac,
        stratify=stratify, random_state=random_state_test)

    if train_frac + test_frac < 1:
        train_idxs, val_idxs = train_test_split(
            _trainval_idxs, train_size=train_frac,
            test_size=1-train_frac-test_frac,
            stratify=stratify[_trainval_idxs] if stratify is not None else None
        )
    else:
        train_idxs = _trainval_idxs
        val_idxs = None
    return train_idxs, val_idxs, test_idxs
