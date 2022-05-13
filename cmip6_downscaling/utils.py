from __future__ import annotations

from hashlib import blake2b

# import dask
# import datatree
# import xarray as xr
# import zarr

# from cmip6_downscaling.methods.common.utils import zmetadata_exists


def str_to_hash(s: str) -> str:
    return blake2b(s.encode(), digest_size=8).hexdigest()


# def write(ds: xr.Dataset | datatree.DataTree, target, use_cache: bool = True) -> str:

#     if use_cache and zmetadata_exists(target):
#         print(f'found existing target: {target}')
#         return target

#     else:
#         print(f'writing target: {target}')
#         out = dask.optimize(ds)[0]
#         if isinstance(ds, xr.Dataset):
#             t = out.to_zarr(target, mode='w', compute=False, consolidated=False)
#             t.compute(retries=5)
#             zarr.consolidate_metadata(target)
#         else:
#             # datatree doesn't support compute=False yet
#             ds.to_zarr(target, mode='w')

#     return target
