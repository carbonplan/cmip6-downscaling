import os

import zarr


def get_store(prefix):
    ''' helper function to create a zarr store'''
    store = zarr.storage.ABSStore(
        'carbonplan-downscaling',
        prefix=prefix,
        account_name="carbonplan",
        account_key=os.environ["BLOB_ACCOUNT_KEY"],
    )
    return store
