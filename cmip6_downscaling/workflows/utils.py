import os

import zarr


def get_store(prefix, account_key=None):
    '''helper function to create a zarr store'''

    if account_key is None:
        account_key = os.environ.get('BLOB_ACCOUNT_KEY', None)

    store = zarr.storage.ABSStore(
        'carbonplan-downscaling',
        prefix=prefix,
        account_name="carbonplan",
        account_key=account_key,
    )
    return store


def generate_batches(n, batch_size, buffer_size, one_indexed=False):
    """
    ds must have a dimension called time that is a valid datetime index 
    """
    # TODO: add tests. if buffer_size == 0, batches == cores 
    # construct 2 test cases 
    
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
