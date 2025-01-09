import numpy as np
# from scipy import stats
# from util.dtype_util import to_iterable, is_iterable
# from numba import njit
# from scipy.signal import welch
# from copy import deepcopy
    

# This only works for 1d array!
# https://stackoverflow.com/questions/30003068/how-to-get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
def unique_with_all_indices_1d(array_1d):

    if array_1d.ndim != 1:
        raise Exception("unique_with_all_indices_1d only works for 1D ndarrays!")

    # creates an array of indices, sorted by unique element
    idx_sort = np.argsort(array_1d)

    # sorts records array so all unique elements are together
    sorted_array_1d = array_1d[idx_sort]

    # returns the unique values, the index of the first occurrence of a value, and the count for each element
    u_vals, idx_start, count = np.unique(sorted_array_1d, return_counts=True, return_index=True)

    # splits the indices into separate arrays
    indices = np.split(idx_sort, idx_start[1:])

    return u_vals, indices


def z_norm_2d_by_row(arr):
    assert arr.ndim == 2
    mean = np.mean(arr, axis=1)
    std_ = np.std(arr, axis=1)
    return (arr - mean[:, np.newaxis]) / std_[:, np.newaxis]
