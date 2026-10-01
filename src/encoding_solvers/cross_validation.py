from pathlib import Path

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


def define_groups(cv_strategy, sub_name, data_dir, space="T1w"):
    """
    Note that "category" will return `incl_labels` corresponding
    to image categories (e.g., 'acorn')
    and "image" will return `incl_labels` corresponding
    to image identities (e.g., 'acorn_01b').

    Note that we are not currently using inner_groups and only separating
    by image type.
    """

    groups = np.loadtxt(
        Path(data_dir, "encoding-inputs", space, f"{sub_name}_stim_labels.txt"),
        dtype=np.str_,
    )
    inner_groups = np.loadtxt(
        Path(data_dir, "encoding-inputs", space, f"{sub_name}_session_labels.txt"),
        dtype=np.str_,
    )

    if cv_strategy == "kfold":
        return None, inner_groups

    if cv_strategy == "image":
        return groups, inner_groups

    if cv_strategy == "category":
        groups = np.asarray([g.rsplit("_", 1)[0] for g in groups])
        return groups, inner_groups

    # FIXME
    # if cv_strategy == "multilabel":
    #     # NOTE : this is consolidating duplicate keys
    #     with open(
    #         Path(
    #             data_dir,
    #             "encoding-inputs",
    #             space,
    #             f"{sub_name}_category53_mapping.json",
    #         )
    #     ) as f:
    #         cat_dict = json.load(f)

    #     cat53_stim_mask_ = [True if g in cat_dict.keys() else False for g in groups]
    #     cat53_X = X_matrix[cat53_stim_mask_]

    #     cat53_dense_labels_ = []
    #     for sv in groups[cat53_stim_mask_]:
    #         cat53_dense_labels_.append(cat_dict.get(sv))

    #     mlb = MultiLabelBinarizer().fit(cat53_dense_labels_)
    #     cat53_y = mlb.transform(cat53_dense_labels_)

    #     X_matrix, y_idx, groups = leave_one_THINGSplus_out(cat53_X, cat53_y)
    #     y_matrix = y_matrix[cat53_stim_mask_][y_idx]
    #     return groups, inner_groups

    ####################################
    # FIXME
    # if average:
    #     # NOTE: shapes hard-coded for three repetitions, 4174 images, THINGS dataset
    #     if groups is not None:
    #         groups = groups[::3]
    #     X_matrix = X_matrix[::3]
    #     y_matrix = np.mean(
    #         y_matrix.reshape(len(X_matrix), 3, y_matrix.shape[-1]), axis=1
    #     )
    ####################################
