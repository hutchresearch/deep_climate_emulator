from copy import deepcopy
import numpy as np


class Normalizer:

    def __init__(self):
        """
        Normalizer constructor. Initializes constants that will be used for
        data transformation.
        """
        self.train_min = 0
        self.train_max = 0
        self.centering_shift_constant = 0
        self.zero_shift_constant = (10 ** -6)

    def transform(self, data, train_len, copy=True):
        """
        Applies log transformation and scales values b/t -1 and 1.

        Args:
            data (ndarray): Collection of data points
            train_len (int): Length of the training set
            copy (bool): If true, creates a copy of th data array

        Returns:
            (ndarray): Array of normalized data points
        """
        if copy:
            data = deepcopy(data)

        # Shift to make all values non-zero
        data += self.zero_shift_constant
        data = np.log2(data)

        # NOTE: We scale relative to the max and min of the training set to
        #       avoid leaking any information from the validation sets.
        self.train_min = data[:train_len].min()
        data -= self.train_min
        self.train_max = data[:train_len].max()
        data /= self.train_max

        # Multiply by 2, and shift values to put between -1 & 1.
        data *= 2
        self.centering_shift_constant = (data.max() - data.min()) / 2
        data -= self.centering_shift_constant

        return data

    def inverse_transform(self, data):
        """
        Applies the inverse transformation.

        Args:
            data (ndarray): Collection of data points

        Returns:
            (ndarray): Array of denormalized data points
        """
        data += self.centering_shift_constant
        data /= 2
        data *= self.train_max
        data += self.train_min
        data = np.power(2, data)
        data -= self.zero_shift_constant

        return data
