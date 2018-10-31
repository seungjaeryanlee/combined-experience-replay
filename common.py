import numpy as np


def get_running_mean(arr, window=30):
    return np.convolve(arr, np.ones((window,)) / window, mode='valid')
