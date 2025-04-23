import numpy as np


def split_data(
    x_train,
    y_train,
    mp_size,
    dp_size,
    rank,
):
    """The function for splitting the dataset uniformly across data parallel groups

    Parameters
    ----------
        x_train : np.ndarray float32
            the input feature of MNIST dataset in numpy array of shape (data_num, feature_dim)

        y_train : np.ndarray int32
            the label of MNIST dataset in numpy array of shape (data_num,)

        mp_size : int
            Model Parallel size

        dp_size : int
            Data Parallel size

        rank : int
            the corresponding rank of the process

    Returns
    -------
        split_x_train : np.ndarray float32
            the split input feature of MNIST dataset in numpy array of shape (data_num/dp_size, feature_dim)

        split_y_train : np.ndarray int32
            the split label of MNIST dataset in numpy array of shape (data_num/dp_size, )

    Note
    ----
        please split the data uniformly across data parallel groups and
        do not shuffle the index as we will shuffle them later
    """

    """TODO: Your code here"""

    # Try to get the correct start_idx and end_idx from dp_size, mp_size and rank and return
    # the corresponding data
    data_size = x_train.shape[0]
    size_per_dp = data_size // dp_size
    dp_index = rank // mp_size
    if dp_index >= dp_size:
        print("Out of bound")
        return None, None
    s_index, e_index = dp_index * size_per_dp, (dp_index + 1) * size_per_dp
    return x_train[s_index: e_index], y_train[s_index: e_index]
