import numpy as np


def leave_one_THINGSplus_out(cat53_X, cat53_y):
    """
    Parameters
    ----------
    sub_name : str
        Subject name
    data_dir : str

    Returns
    -------
    X : np.arr
    y : np.arr
    groups : np.arr

    Note
    ----
    The resulting train, test splits will be of unequal sizes ;
    that is, images that are not labelled "animal" may also be not labelled
    "breakfast food," and so assigned to the training split multiple times.
    The resulting distribution of labels is known to have a significant
    rightward-skew given the pre-existing label distribution (i.e., the category
    "animal" is more likely to occur overall).
    """
    groups = []
    X = []
    y_idx = []
    for grp_lbl in range(53):
        for idx, (y_, X_) in enumerate(zip(cat53_y, cat53_X)):
            if y_[grp_lbl] == 1:
                X.append(X_)
                y_idx.append(idx)
                groups.append(grp_lbl + 1)

    X = np.asarray(X)
    y_idx = np.asarray(y_idx)
    groups = np.asarray(groups)

    return X, y_idx, groups
