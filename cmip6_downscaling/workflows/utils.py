import os

import zarr


def get_store(key):
    ''' helper function to create a zarr store'''
    store = zarr.storage.ABSStore(
        'carbonplan-scratch',
        prefix=f'regridded-cmip-data/{key}',
        account_name="carbonplan",
        account_key=os.environ["BLOB_ACCOUNT_KEY"],
    )
    return store
