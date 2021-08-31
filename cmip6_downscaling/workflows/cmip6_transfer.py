import pandas as pd
import json
import os
import adlfs
import fsspec
import intake
import xarray as xr
from prefect import Flow, Parameter, task
from prefect.run_configs import KubernetesRun
from prefect.storage import GCS
import datetime
from datetime import timedelta


variable_ids = ["pr", "tasmin", "tasmax"]
col_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
connection_string = "BlobEndpoint=https://cmip6downscaling.blob.core.windows.net/;QueueEndpoint=https://cmip6downscaling.queue.core.windows.net/;FileEndpoint=https://cmip6downscaling.file.core.windows.net/;TableEndpoint=https://cmip6downscaling.table.core.windows.net/;SharedAccessSignature=sv=2020-08-04&ss=bfqt&srt=co&sp=rwdlacuptfx&se=2022-01-02T04:42:34Z&st=2021-08-31T19:42:34Z&spr=https&sig=lx2ZwVnXnTZGdykX9kjMrE%2F6c%2FyKL0HnEwxrGHyxbyo%3D"
csv_catalog_path = "az://cmip6/az_cmip6.csv"
json_catalog_path = "az://cmip6/az_cmip6.json"

##### Helper Functions ####


def open_json_catalog():
    data = json.load(open("cmip_catalog.json"))
    return data


def write_json_catalog_to_azure():
    data = open_json_catalog()
    with fsspec.open(json_catalog_path, "w", connection_string=connection_string) as f:
        json.dump(data, f)


# def read_json_catalog_on_azure():
#     with fsspec.open(json_catalog_path, "r", connection_string=connection_string) as f:


def create_catalog_df():
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
    df = create_catalog_df()
    df.to_csv(
        csv_catalog_path,
        storage_options={"connection_string": connection_string},
        index=False,
    )
    return df


def load_csv_catalog():
    try:
        df = pd.read_csv(
            csv_catalog_path, storage_options={"connection_string": connection_string}
        )
    except:
        df = save_empty_catalog()
    return df


def rename_gs_to_az(src):
    tgt = "az://cmip6/" + src.split("CMIP6/")[1]
    return tgt


def zarr_is_complete(store, check=".zmetadata"):
    """Return true if Zarr store is complete"""
    return check in store


#### Tasks #####


@task
def write_csv_catalog(df):
    df.to_csv(
        csv_catalog_path,
        storage_options={"connection_string": connection_string},
        index=False,
    )


@task
def append_to_catalog_df(src_chunk, df):
    df = df.append(src)
    return df


@task(nout=2)
def map_src_tgt(src, tgt, connection_string):
    src_map = fsspec.get_mapper(src)
    tgt_map = fsspec.get_mapper(tgt, connection_string=connection_string)
    return src_map, tgt_map


@task
def check_target_store(tgt_map):
    check_status = zarr_is_complete(tgt_map)
    return check_status


@task()
def load_zarr_store(src_map):
    xdf = xr.open_zarr(src_map, decode_cf=False, consolidated=True)
    return xdf


@task()
def process_data(xdf):
    # do any data processing steps...
    cleaned_xdf = xdf
    return cleaned_xdf


@task()
def copy_cleaned_data(cleaned_xdf, tgt_map, overwrite=False):
    cleaned_xdf.to_zarr(tgt_map, consolidated=True)


# @task(nout=2)
def retrive_cmip6_catalog():
    col = intake.open_esm_datastore(col_url)

    # get all possible simulations
    full_subset = col.search(
        activity_id=["CMIP", "ScenarioMIP"],
        experiment_id=["historical", "ssp245", "ssp370", "ssp585"],
        table_id="Amon",
        grid_label="gn",
        variable_id=variable_ids,
    )

    # get historical simulations
    hist_subset = full_subset.search(
        activity_id=["CMIP"],
        experiment_id=["historical"],
        require_all_on=["variable_id"],
    )

    # get future simulations
    ssp_subset = full_subset.search(
        activity_id=["ScenarioMIP"],
        experiment_id=["ssp245", "ssp370", "ssp585"],
        require_all_on=["variable_id"],
    )
    return hist_subset, ssp_subset


"""testing transfer"""
hist_subset, ssp_subset = retrive_cmip6_catalog()


src = ssp_subset.df.zstore.iloc[0]
tgt = "az://cmip6/" + src.split("CMIP6/")[1]
src_map = fsspec.get_mapper(src)

tgt_map = fsspec.get_mapper(tgt, connection_string=connection_string)
# xdf = xr.open_zarr(src_map, decode_cf=False, consolidated=True)


run_config = KubernetesRun(
    cpu_request=2,
    memory_request="2Gi",
    image="gcr.io/carbonplan/hub-notebook:b2419ff",
    labels=["gcp-us-central1-b"],
    env={"TZ": "UTC"},
)
storage = GCS("carbonplan-prefect")


# with Flow(
#     name="Transfer CMIP6 from gs to az", storage=storage, run_config=run_config,
# ) as flow:
with Flow(name="Transfer CMIP6 from gs to az") as flow:
    # 0.5
    hist_subset, ssp_subset = retrive_cmip6_catalog()
    src_test_list = [
        "gs://cmip6/CMIP6/ScenarioMIP/MRI/MRI-ESM2-0/ssp245/r1i1p1f1/Amon/pr/gn/v20190222/"
    ]
    # loop though ssp_subset/hist_subset.df
    for row in ssp_subset.df.iterrows():
        src = row["zstore"]
        tgt = rename_gs_to_az(src)
        # 1. attempt to map src and target (what to do if fails.. retry?). if fails, append to dict of failed map, pass other flows
        src_map, tgt_map = map_src_tgt(src, tgt, connection_string)
        # 2. check that store does not exist, if exists already, pass
        check_status = check_target_store(tgt_map)
        if check_status == False:
            missed = []
            try:
                # 3. load store
                xdf = load_zarr_store(src_map)
                # 4. clean store
                cleaned_xdf = process_data(xdf)
                # 5. export store
                copy_cleaned_data(cleaned_xdf, tgt_map)
                # 6. update catalog
                df = append_to_catalog_df(row, df)
            except:
                missed.append(src)

    write_csv_catalog(df)


# if __name__ == "__main__":
#     # flow = prefect_flow()
#     # flow.register(project_name="offset-fires")
#     flow.run()
