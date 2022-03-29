from __future__ import annotations

import fsspec
from prefect import task

from .. import config


def instantiate_az_filesystem() -> fsspec.filesystem:
    """Returns azure filesystem using fsspec

    Returns:
       fsspec.filesystem object : Azure filesystem
    """
    # should this have another more generalized conn str?
    fs = fsspec.filesystem(
        'az', connection_string=config.get("storage.top_level.storage_options.connection_string")
    )
    return fs


def remove_stores(fpaths: list[str]):
    """Removes AZ store

    Args:
        fpath (str): Valid Azure fpath

    """
    fs = instantiate_az_filesystem()
    for fil in fpaths:
        try:
            fs.rm(fil, recursive=True)
        except Exception as e:
            print(e)
            print('File not found: ', fil)


@task(log_stdout=True)
def run_rsfip(gcm_identifier: str, obs_identifier: str):
    print(gcm_identifier)
    # remove_stores_from_input_params will stall if a task decorator is added. For some reason this seems to work...
    remove_stores_from_input_params(gcm_identifier, obs_identifier)


def remove_stores_from_input_params(gcm_identifier: str, obs_identifier: str):
    """Removes any stores in intermediate/results subdirectories of flow-ouputs based on input parameter strings.

    Args:
        gcm_identifier (str): gcm_identifier string from path_builder_task output
        obs_identifier (str): obs_identifier string from path_builder_task output
    """

    print(
        '\n -------------------------------------------------------\n',
        '-------------------------------------------------------\n',
        '------ CLEANUP FLAG ENABLED. CACHING IS DISABLED ------\n',
        '-------------------------------------------------------\n',
        '-------------------------------------------------------\n',
    )

    for stage in config.get("methods.bcsd.process_stages"):
        prefix_list = config.get(f"methods.bcsd.process_stages.{stage}")
        stage_uri = config.get(f"storage.{stage}.uri")
        for prefix in prefix_list:
            path_template = config.get(
                f"methods.bcsd.process_stages.{stage}.{prefix}.path_template"
            )
            path_template = path_template.format(
                gcm_identifier=gcm_identifier, obs_identifier=obs_identifier
            )
            store_uri = stage_uri + path_template
            store_xpersist_cache = (
                stage_uri + config.get("storage.xpersist_store_name") + path_template
            )
            try:
                print('Removing: ', store_uri, '\n', store_xpersist_cache, '\n')
                remove_stores([store_uri, store_xpersist_cache])
            except Exception as e:
                print(e)
                print('Unable to remove: ', store_uri)
