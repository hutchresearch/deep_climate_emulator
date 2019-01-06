import numpy as np


def compute_mse(actual, forecast):
    """
    Computes the Mean Squared Error between the tensors 'actual' and
    'forecast', representing the ground truth outcomes and model outcomes,
    respectively.

    Args:
        actual (ndarray): Ground truth outcome
        forecast (ndarray): Model outcome

    Returns:
        (float): Mean squared error
    """
    return np.mean(np.square(np.subtract(forecast, actual)))
