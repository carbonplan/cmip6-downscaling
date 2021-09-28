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
