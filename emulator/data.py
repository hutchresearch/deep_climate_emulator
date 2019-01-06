import os
import numpy as np


def convert_to_mm_per_day(maps, units='kg m-2 s-1'):
    """
    Convert precipitation maps to mm/day

    Returns:
        (ndarray): Converted precipitation field
    """
    if units == 'kg m-2 s-1':
        return np.multiply(maps, 86400)
    else:
        raise ValueError('Conversion for units=%s not supported' % units)


def load_area_weights():
    """
    Load cosine of latitude weights from CSV file.

    Returns:
        (ndarray): Cosine of latitude weights (64 x 128)
    """
    pkg_dir = os.path.dirname(__file__)
    return np.loadtxt(open("%s/data/latitude_cosines.csv" % pkg_dir, "rb"),
                      delimiter=",", dtype=float)


def create_split_bounds(N, train_pct):
    """
    Computes split bounds for train, dev, and test.

    Args:
        N (int): Number of data points in the time series
        train_pct (float): Percent of data to be used for the training set

    Returns:
        (int): Length of the training set
        (int): Length of the dev set
        (int): Length of the test set
    """
    train_len = int(round(train_pct * N))
    if ((N - train_len) % 2) != 0:
        train_len += 1

    # NOTE: We're assume the dev and test set are equal in length.
    test_len = dev_len = int((N - train_len) / 2)

    assert "Not all data points are being used. Check create_split_bounds()", \
        (train_len + test_len + dev_len) == N

    return train_len, dev_len, test_len


def train_dev_test_split(data, train_pct=0.7):
    """
    Split data into train, dev, and test sets.

    Args:
        data (ndarray): N-dimensional array containing all data points
        train_pct (float): Percent of data to be used for training

    Returns:
        train (ndarray): Training set
        dev (ndarray): Dev set
        test (ndarray): Test set

    """
    train_len, dev_len, test_len = create_split_bounds(len(data), train_pct)

    # Train (70%)
    train = data[0:train_len]

    # Dev (15%)
    dev_ub = (train_len + dev_len)
    dev = data[train_len:dev_ub]

    # Test (15%)
    test = data[dev_ub:]

    assert "One of the sets contains an unexpected number of elements", \
        (len(train) == train_len and len(dev) == dev_len and len(test) == test_len)

    return train, dev, test
