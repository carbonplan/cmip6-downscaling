#!/usr/bin/env python
import os

import fsspec
import pandas as pd
from dask.distributed import Client

from cmip6_downscaling.data.cmip import cmip
from cmip6_downscaling.workflows.utils import get_store

target = 'cmip6/raw/conus/monthly/{key}.zarr'
max_members = 5
skip_existing = False


def slim_cmip_key(key, member_id):
    _, _, source_id, experiment_id, _, _ = key.split('.')
    out_key = f'{source_id}.{experiment_id}.{member_id}'
    return out_key


def main():

    model_dict, data = cmip()

    written_keys = []
    for full_key, ds in data.items():

        valid_members = 0

        for member_id in ds.member_id.values:

            # only extract `max_members` members (at most)
            if valid_members >= max_members:
                break

            # get the output zarr store
            member_key = slim_cmip_key(full_key, member_id)
            prefix = target.format(key=member_key)
            store = get_store(prefix)
            print(prefix)

            # extract a single member and rechunk
            member_ds = ds.sel(member_id=member_id).chunk({'lat': -1, 'lon': -1, 'time': 198})

            # check that there is data for the full record
            if (
                member_ds.isel(lon=0, lat=0)
                .isnull()
                .any()
                .to_array(name='variables')
                .any()
                .load()
                .item()
            ):
                print('--> skipping, missing some data')
                store.clear()
                continue

            # clean encoding
            for v in member_ds:
                if 'chunks' in member_ds[v].encoding:
                    del member_ds[v].encoding['chunks']

            # write store
            if skip_existing and '.zmetadata' in store:
                print('++++ skipping write', prefix)
            else:
                store.clear()
                member_ds.to_zarr(store, consolidated=True, mode='w')
            valid_members += 1
            written_keys.append(prefix)

    d = {}
    for k in written_keys:
        if 'historical' in k:
            if k not in d:
                d[k] = False
        else:
            pieces = k.split('.')
            pieces[1] = 'historical'
            k2 = '.'.join(pieces)
            if k2 in written_keys:
                d[k2] = True
                d[k] = True
            else:
                d[k] = False

    df = (
        pd.DataFrame.from_dict(d, orient='index')
        .reset_index()
        .rename(columns={0: 'has_match', 'index': 'path'})
    )
    for i, row in df.iterrows():
        model, scenario, member, _ = row.path.split('/')[-1].split('.')

        df.loc[i, 'model'] = model
        df.loc[i, 'scenario'] = scenario
        df.loc[i, 'member'] = member

    with fsspec.open(
        'az://carbonplan-downscaling/cmip6/ssps_with_matching_historical_members.csv',
        'w',
        account_name='carbonplan',
        account_key=os.environ['BLOB_ACCOUNT_KEY'],
    ) as f:
        df.to_csv(f)


if __name__ == '__main__':
    client = Client(n_workers=20)
    print(client)
    print(client.dashboard_link)

    main()
