import numpy as np


def generate_batches(n, batch_size, buffer_size, one_indexed=False):
    """
    Given the max value n, batch_size, and buffer_size, returns batches (include the buffer) and
    cores (exclude the buffer). For the smallest numbers, the largest values would be included in the buffer, and
    vice versa. For example, with n=10, batch_size=5, buffer_size=3, one_indexed=False. The `cores` output will contain
    [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], and `batches` output will contain [[7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2]].
    
    Parameters
    ----------
    n : int
        The max value to be included.
    batch_size : int
        The number of core values to include in each batch.
    buffer_size : int
        The number of buffer values to include in each batch in both directions.
    one_indexed : bool
        Whether we should consider n to be one indexed or not. With n = 2, one_indexed=False would generate cores containing [0, 1].
        One_indexed=True would generate cores containing [1, 2].
    
    Returns
    -------
    batches : List
        List of batches including buffer values.
    cores : List
        List of core values in each batch excluding buffer values.
    """
    cores = []
    batches = []
    if one_indexed:
        xmin = 1
        xmax = n + 1
    else:
        xmin = 0
        xmax = n
    for start in range(xmin, xmax, batch_size):
        end = min(start + batch_size, xmax)
        cores.append(np.arange(start, end))

        # add buffer
        end = end + buffer_size
        start = start - buffer_size
        batch = np.arange(start, end)
        batch[batch < xmin] += n
        batch[batch > xmax - 1] -= n
        batches.append(batch)

    return batches, cores