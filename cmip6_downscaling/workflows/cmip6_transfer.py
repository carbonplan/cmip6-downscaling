# Imports -----------------------------------------------------------
import json
import os

import fsspec
import intake
import pandas as pd
import xarray as xr
from prefect import Flow, task

from .. import runtimes

# vars/pathing -----------------------------------------------------------

# variable_ids = ["pr", "tasmin", "tasmax"]
runtime = runtimes.get_runtime()
variable_ids = ["ua", "va"]
col_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
csv_catalog_path = "az://cmip6/pangeo-cmip6.csv"
json_catalog_path = "az://cmip6/pangeo-cmip6.json"

# Helper Functions -----------------------------------------------------------


def open_json_catalog():
    """Loads local CMIP6 JSON intake catalog"""
    data = json.load(open("cmip_catalog.json"))
    return data


def write_json_catalog_to_azure():
    """Writes json catalog to Azure store"""
    data = open_json_catalog()
    with fsspec.open(json_catalog_path, "w", connection_string=connection_string) as f:
        json.dump(data, f)


def create_catalog_df():
    """Creates an empty DataFrame for a catalog"""
    df = pd.DataFrame(
        columns=[
            "activity_id",
            "institution_id",
            "source_id",
            "experiment_id",
            "member_id",
            "table_id",
            "variable_id",
            "grid_label",
            "zstore",
            "dcpp_init_year",
            "version",
        ]
    )
    return df


def save_empty_catalog():
    """Saves initial catalog"""
    df = create_catalog_df()
    df.to_csv(
        csv_catalog_path,
        storage_options={"connection_string": connection_string},
        index=False,
    )
    return df


def load_csv_catalog():
    """Loads existing csv catalog, if catalog is missing creates an empty one"""
    try:
        df = pd.read_csv(csv_catalog_path, storage_options={"connection_string": connection_string})
    except:
        df = save_empty_catalog()
    return df


def rename_gs_to_az(src):
    """String formatting to update the prefix of the CMIP6 gs store location to Azure"""
    tgt = "az://cmip6/" + src.split("CMIP6/")[1]
    return tgt


def zarr_is_complete(store, check=".zmetadata"):
    """Return true if Zarr store is complete"""
    return check in store


def write_csv_catalog(df):
    """Writes modified csv catalog to Azure storage"""
    df.to_csv(
        csv_catalog_path,
        storage_options={"connection_string": connection_string},
        index=False,
    )


def map_src_tgt(src, tgt, connection_string):
    """Uses fsspec to creating mapped object from source and target connection strings"""
    src_map = fsspec.get_mapper(src)
    tgt_map = fsspec.get_mapper(tgt, connection_string=connection_string)
    return src_map, tgt_map


def load_zarr_store(src_map):
    """Given a mapped source, loads Zarr store and returns xarray dataset"""
    xdf = xr.open_zarr(src_map, decode_cf=False, consolidated=True)
    return xdf


def copy_cleaned_data(xdf, tgt_map, overwrite=True):
    """Copies xarray dataset to zarr store for given target"""
    if overwrite is True:
        mode = "w"
    else:
        mode = "r"
    xdf.to_zarr(tgt_map, mode=mode, consolidated=True)


def retrive_cmip6_catalog():
    """Returns historical and scenario results as intake catalogs"""
    col = intake.open_esm_datastore(col_url)

    # get all possible simulations
    full_subset = col.search(
        activity_id=["CMIP", "ScenarioMIP"],
        experiment_id=["historical", "ssp245", "ssp370", "ssp585"],
        member_id="r1i1p1f1",
        table_id="day",
        grid_label="gn",
        variable_id=variable_ids,
    )

    return full_subset  # hist_subset, ssp_subset


# Prefect Task(s) -----------------------------------------------------------


@task
def copy_to_azure(src_tgt_uris):
    src_uri, tgt_uri = src_tgt_uris
    src_map, tgt_map = map_src_tgt(src_uri, tgt_uri, connection_string)
    if not zarr_is_complete(tgt_map):
        xdf = load_zarr_store(src_map)
        copy_cleaned_data(xdf, tgt_map)


# Prefect Flow -----------------------------------------------------------


with Flow(
    name="Transfer_CMIP6",
    storage=runtime.storage,
    run_config=runtime.run_config,
    executor=runtime.executor,
) as flow:
    full_subset = retrive_cmip6_catalog()
    df = load_csv_catalog()
    subset_df = full_subset.df
    uris = [(src_uri, rename_gs_to_az(src_uri)) for src_uri in subset_df.zstore.to_list()]
    df_new = subset_df.copy()
    copy_to_azure.map(uris)
    df_new["zstore"] = [tgt_uri[1] for tgt_uri in uris]
    updated_df = pd.concat([df, df_new], axis=0).drop_duplicates(keep="last", ignore_index=True)
    updated_df.to_csv(
        csv_catalog_path, index=False, storage_options={"connection_string": connection_string}
    )
