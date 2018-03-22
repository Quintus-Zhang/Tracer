import numpy as np


def num_to_idx(value, grid):
    """

    :param value:
    :param grid:
    :return:
    """
    if value > grid[-1]:
        value = grid[-1]
    if value < grid[0]:
        value = grid[0]

    step = (grid[-1] - grid[0]) / (len(grid) - 1)

    idx = int((value - grid[0]) / step)
    return idx
