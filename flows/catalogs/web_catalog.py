import datetime
import functools
import itertools
import json
import random
import traceback

import fsspec
import intake
import packaging.version
from prefect import Flow, Parameter, task
from upath import UPath

from cmip6_downscaling import config, runtimes

config.set(
    {
        'runtime.cloud.extra_pip_packages': 'git+https://github.com/carbonplan/cmip6-downscaling.git git+https://github.com/intake/intake-esm.git git+https://github.com/ncar-xdev/ecgtools.git'
    }
)

runtime = runtimes.CloudRuntime()


def get_license(source_id: str, derived_product: bool = False) -> dict:
    """
    Get the license.

    Parameters
    ----------
    source_id : str
        The source id.
    derived_product : bool, optional
        Whether the product is derived. The default is False.

    Returns
    -------
    dict
        The license."""

    LICENSE_MAPPING = {
        'ACCESS-CM2': '',
        'ACCESS-ESM1-5': 'CC-BY-4.0',
        'AWI-CM-1-1-MR': 'CC-BY-4.0',
        'AWI-ESM-1-1-LR': 'CC-BY-4.0',
        'BCC-CSM2-MR': 'CC-BY-SA-4.0',
        'BCC-ESM1': 'CC-BY-SA-4.0',
        'CAMS-CSM1-0': 'CC-BY-SA-4.0',
        'CAS-ESM2-0': '',
        'CESM2': 'CC-BY-4.0',
        'CESM2-FV2': 'CC-BY-4.0',
        'CESM2-WACCM': 'CC-BY-4.0',
        'CESM2-WACCM-FV2': 'CC-BY-4.0',
        'CMCC-CM2-HR4': 'CC-BY-4.0',
        'CMCC-CM2-SR5': 'CC-BY-4.0',
        'CMCC-ESM2': 'CC-BY-4.0',
        'CNRM-CM6-1': 'CC-BY-4.0',
        'CNRM-CM6-1-HR': 'CC-BY-4.0',
        'CNRM-ESM2-1': 'CC-BY-4.0',
        'CanESM5': 'CC-BY-SA-4.0',
        'CanESM5-CanOE': 'CC-BY-SA-4.0',
        'EC-Earth3': 'CC-BY-4.0',
        'EC-Earth3-AerChem': 'CC-BY-4.0',
        'EC-Earth3-CC': 'CC-BY-4.0',
        'EC-Earth3-Veg': 'CC-BY-4.0',
        'EC-Earth3-Veg-LR': 'CC-BY-4.0',
        'FGOALS-g3': 'CC-BY-SA-4.0',
        'FIO-ESM-2-0': 'CC-BY-SA-4.0',
        'GFDL-CM4': 'CC-BY-4.0',
        'GISS-E2-1-G': 'CC0-1.0',
        'GISS-E2-1-G-CC': 'CC0-1.0',
        'GISS-E2-1-H': 'CC0-1.0',
        'HadGEM3-GC31-LL': 'CC-BY-4.0',
        'HadGEM3-GC31-MM': 'CC-BY-4.0',
        'ICON-ESM-LR': 'CC-BY-4.0',
        'IITM-ESM': 'CC-BY-SA-4.0',
        'INM-CM4-8': 'CC-BY-SA-4.0',
        'INM-CM5-0': 'CC-BY-SA-4.0',
        'IPSL-CM6A-LR': 'CC-BY-4.0',
        'KACE-1-0-G': 'CC-BY-SA-4.0',
        'MCM-UA-1-0': '',
        'MIROC-ES2H': 'CC-BY-4.0',
        'MIROC-ES2L': 'CC-BY-4.0',
        'MIROC6': 'CC-BY-4.0',
        'MPI-ESM-1-2-HAM': 'CC-BY-SA-4.0',
        'MPI-ESM1-2-HR': 'CC-BY-4.0',
        'MPI-ESM1-2-LR': 'CC-BY-SA-4.0',
        'MRI-ESM2-0': 'CC-BY-4.0',
        'NESM3': 'CC-BY-SA-4.0',
        'NorCPM1': 'CC-BY-4.0',
        'NorESM2-LM': 'CC-BY-4.0',
        'NorESM2-MM': 'CC-BY-4.0',
        'SAM0-UNICON': 'CC-BY-SA-4.0',
        'TaiESM1': 'CC-BY-4.0',
        'UKESM1-0-LL': 'CC-BY-4.0',
    }

    license = LICENSE_MAPPING.get(source_id)

    if derived_product and license in {'CC0-1.0', 'CC-BY-4.0'}:
        # For derived products (pyramids, and downscaled data) generated
        # from CMIP6 data with permissive licenses,
        # we promote the license to CC-BY-4.0
        license = 'CC-BY-4.0'

    if license:
        licences = {
            'CC-BY-4.0': {
                'name': 'CC-BY-4.0',
                'url': 'https://creativecommons.org/licenses/by/4.0/',
            },
            'CC-BY-SA-4.0': {
                'name': 'CC-BY-SA-4.0',
                'url': 'https://creativecommons.org/licenses/by-sa/4.0/',
            },
            'CC0-1.0': {
                'name': 'CC0-1.0',
                'url': 'https://creativecommons.org/publicdomain/zero/1.0/',
            },
        }

        return licences[license]
    else:
        return {}


def parse_cmip6(store: str, root_path: str) -> dict[str, str]:

    from ecgtools.builder import INVALID_ASSET, TRACEBACK

    from cmip6_downscaling.methods.common.utils import zmetadata_exists

    try:

        path = UPath(store)
        if not zmetadata_exists(path):
            raise ValueError(f'{path} not a valid zarr store')

        aggregation = None
        name = path.name
        (
            activity_id,
            institution_id,
            source_id,
            experiment_id,
            table_id,
            grid_label,
            member_id,
            variable_id,
            timescale,
        ) = name.split('.')
        query = {
            'activity_id': activity_id,
            'institution_id': institution_id,
            'source_id': source_id,
            'experiment_id': experiment_id,
            'table_id': table_id,
            'member_id': member_id,
            'variable_id': variable_id,
        }
        if timescale == '1MS':
            timescale = 'month'
        elif timescale == 'YS':
            timescale = 'year'

        if timescale in {'year', 'month'}:
            if variable_id in {'tasmin', 'tasmax'}:
                aggregation = 'mean'
            elif variable_id in {'pr'}:
                aggregation = 'sum'

        uri = f"{root_path}/{str(path).split('//')[-1]}"
        cat_url = "https://cmip6downscaling.blob.core.windows.net/cmip6/pangeo-cmip6.json"
        cat = intake.open_esm_datastore(cat_url)
        stores = cat.search(**query).df.zstore.tolist()
        if stores:
            original_dataset_uris = [f"{root_path}/{dataset.split('//')[-1]}" for dataset in stores]
        else:
            original_dataset_uris = []
        return {
            'name': name,
            **query,
            'timescale': timescale,
            'aggregation': aggregation,
            'uri': uri,
            'original_dataset_uris': original_dataset_uris,
            'method': 'Raw',
            'license': get_license(source_id),
        }

    except Exception:
        return {INVALID_ASSET: path, TRACEBACK: traceback.format_exc()}


@task(log_stdout=True)
def get_cmip6_pyramids(paths: list[str], cdn: str, root_path: str) -> list[dict]:
    """
    Get the pyramids for a list of CMIP6 stores.

    Parameters
    ----------
    paths : list[str]
        List of CMIP6 stores.
    cdn : str
        CDN to use for the pyramids.
    root_path : str
        Root path for the pyramids.

    Returns
    -------
    list[dict]
        List of pyramids.
    """
    import ecgtools

    builder = ecgtools.Builder(
        paths=paths,
        depth=1,
        joblib_parallel_kwargs={'n_jobs': -1, 'verbose': 1},
        exclude_patterns=["*.json"],
    )
    builder.build(parsing_func=functools.partial(parse_cmip6, root_path=root_path))

    return builder.df.to_dict(orient='records')


def construct_destination_name_and_path(
    *,
    activity_id: str,
    institution_id: str,
    source_id: str,
    experiment_id: str,
    member_id: str,
    variable_id: str,
    downscaling_method: str,
    root_path: str,
) -> dict:
    """
    Construct the destination name and path for the CMIP6 pyramids.

    """

    destination_name = f'{activity_id}.{institution_id}.{source_id}.{experiment_id}.{member_id}.day.{downscaling_method}.{variable_id}'
    destination_path = f'az://version1/data/{downscaling_method}/{destination_name}.zarr'

    return {
        'destination_name': destination_name,
        'destination_path': from_az_to_https(destination_path, root_path),
    }


def from_az_to_https(uri: str, root: str) -> str:
    """
    Convert an Azure blob URI to a HTTPS URI.

    """
    return f"{root}/{uri.split('//')[-1]}" if uri else None


def parse_cmip6_downscaled_pyramid(
    data, cdn: str, root_path: str, derived_product: bool = True
) -> list[dict]:

    """
    Parse metadata for given CMIP6 downscaled pyramid.

    Parameters
    ----------
    data : str
        Prefect run metadata.
    cdn : str
        CDN to use for the data.
    root_path : str
        Root path for the data.
    derived_product : bool, optional
        Whether the data is a derived product.
        Defaults to True.

    Returns
    -------
    list[dict]
        List of dictionaries with the metadata for the data."""

    parameters = data['parameters']
    datasets = data['datasets']

    query = {
        'source_id': parameters['model'],
        'member_id': parameters['member'],
        'experiment_id': 'historical'
        if parameters['scenario'] == 'hist'
        else parameters['scenario'],
        'variable_id': parameters['variable'],
    }

    cat_url = "https://cmip6downscaling.blob.core.windows.net/cmip6/pangeo-cmip6.json"
    cat = intake.open_esm_datastore(cat_url).search(table_id='day', **query)
    records = cat.df.to_dict(orient='records')
    if records:
        entry = records[0]
        query['institution_id'] = entry['institution_id']
        query['activity_id'] = entry['activity_id']
        query['original_dataset_uris'] = [
            f"{from_az_to_https(str(entry['zstore']), root_path)}" for entry in records
        ]

    method_mapping = {
        'deepsd': 'DeepSD',
        'raw': 'Raw',
        'maca': 'MACA',
        'gard': 'GARD-SV',
        'gard_multivariate': 'GARD-MV',
    }

    method = method_mapping[parameters['method']]

    template = {
        **query,
        'method': method,
        'license': get_license(query['source_id'], derived_product=derived_product),
        'cmip6_downscaling_version': data.get('attrs', {}).get('version'),
    }

    if parameters['method'] == 'maca':
        template['daily_downscaled_data_uri'] = datasets.get('final_bias_corrected_full_time_path')

    elif parameters['method'] in {'gard', 'gard_multivariate'}:
        template['daily_downscaled_data_uri'] = datasets.get('model_output_path')

    results = []
    if parameters['method'] != 'deepsd':
        options = {'daily', 'monthly', 'annual'}
        timescales = {'daily': 'day', 'monthly': 'month', 'annual': 'year'}
        for option in options:
            key = f'{option}_pyramid_path'
            if key in datasets:
                template['timescale'] = timescales[option]
                attributes = {
                    **template,
                    **{
                        'uri': f"{from_az_to_https(str(datasets[key]), cdn)}",
                        'name': UPath(datasets[key]).name,
                    },
                }
                results.append(attributes)

    else:
        _methods = ['raw', 'bias_corrected']
        _pyramids = ['monthly_pyramid_path', 'annual_pyramid_path']
        _options = itertools.product(_methods, _pyramids)
        _options = ['_'.join(option) for option in _options]

        for option in _options:
            if option in datasets:
                if 'bias_corrected' in option:
                    template['method'] = f'{method}-BC'
                    template['daily_downscaled_data_uri'] = datasets.get(
                        'bias_corrected_shifted_model_output_path'
                    )
                else:
                    template['daily_downscaled_data_uri'] = datasets.get(
                        'shifted_model_output_path'
                    )

                if 'month' in option:
                    template['timescale'] = 'month'
                elif 'annual' in option:
                    template['timescale'] = 'year'
                pyramid_attrs = {
                    **template,
                    **{
                        'uri': f"{from_az_to_https(str(datasets[option]), cdn)}",
                        'name': UPath(datasets[option]).name,
                    },
                }
                results.append(pyramid_attrs)

    for attrs in results:
        if attrs['timescale'] in {'year', 'month'}:
            if attrs['variable_id'] in {'tasmin', 'tasmax'}:
                attrs['aggregation'] = 'mean'
            elif attrs['variable_id'] in {'pr'}:
                attrs['aggregation'] = 'sum'

        targets = construct_destination_name_and_path(
            activity_id=attrs['activity_id'],
            institution_id=attrs['institution_id'],
            source_id=attrs['source_id'],
            experiment_id=attrs['experiment_id'],
            member_id=attrs['member_id'],
            variable_id=attrs['variable_id'],
            downscaling_method=attrs['method'],
            root_path=root_path,
        )
        attrs[
            'name'
        ] = f"{targets['destination_name']}.{attrs['timescale']}"  # make sure the name is unique
        attrs['destination_path'] = targets['destination_path']
        attrs['source_path'] = from_az_to_https(attrs['daily_downscaled_data_uri'], root_path)

    return results


def pick_latest_version(results: list[str]) -> list[str]:
    """
    Pick the latest version.

    Parameters
    ----------
    results : list[str]
        List of results.

    Returns
    -------
    list[str]
        List of results.
    """

    def parse(path: str):
        parts = path.split('/')
        return {'version': packaging.version.Version(parts[-4]), 'param': parts[-2], 'run': path}

    items = [parse(path) for path in results]
    groups = itertools.groupby(items, key=lambda item: item['param'])
    latest_results = []
    for key, group in groups:
        group = list(group)
        latest = max(group, key=lambda item: item['version'])
        latest_results.append(latest['run'])

    print(f'{len(latest_results)} of {len(results)} results are kept')
    return latest_results


def filter_version_results(
    *, minimum_version: str, maximum_version: str, exclude_local_version: bool, results: list[str]
) -> list[str]:

    """
    Filter the results by version.

    Parameters
    ----------
    minimum_version : str
        The minimum version to keep.
    maximum_version : str
        The maximum version to keep.
    exclude_local_version : bool
        Whether to exclude the local version.
    results : list[str]
        The results to filter.

    Returns
    -------
    list[str]
        The filtered results.
    """
    minimum_version = packaging.version.Version(minimum_version)
    if maximum_version:
        maximum_version = packaging.version.Version(maximum_version)

    valid_results = []
    invalid_results = []

    for result in results:
        version = result.split("/")[-4]
        print(f"Checking version {version}")
        version = packaging.version.Version(version)

        if (
            (maximum_version is None or version <= maximum_version)
            and version >= minimum_version
            and (not exclude_local_version or version.local is None)
        ):
            valid_results.append(result)

        else:
            invalid_results.append(result)

    print(f"{len(valid_results)} of {len(results)} results are kept")
    print(f"{len(invalid_results)} of {len(results)} results are discarded")
    print("\n***** Invalid results: *****\n")
    for item in sorted(random.sample(invalid_results, 10)):
        print(item)
    print("\n***** End of invalid results *****\n")
    return valid_results


@task(log_stdout=True)
def get_cmip6_downscaled_pyramids(
    *,
    path,
    cdn: str,
    root_path: str,
    minimum_version: str,
    maximum_version: str = None,
    exclude_local_version: bool = True,
):

    """
    Get CMIP6 downscaled pyramids.

    Parameters
    ----------
    path : str
        Path to the CMIP6 downscaled pyramids.
    cdn : str
        CDN URL.
    root_path : str
        Root path of the CMIP6 downscaled pyramids.
    minimum_version : str
        Minimum version of the cmip6-downscaling package.
    maximum_version : str, optional
        Maximum version of the cmip6-downscaling package. The default is None.
    exclude_local_version : bool, optional
        Exclude local version of the cmip6-downscaling package. The default is True.

    Returns
    -------
    list[dict]
        List of CMIP6 downscaled pyramids.
    """

    mapper = fsspec.get_mapper(path)
    fs = mapper.fs

    latest_runs = fs.glob(path)
    latest_runs = filter_version_results(
        minimum_version=minimum_version,
        maximum_version=maximum_version,
        exclude_local_version=exclude_local_version,
        results=latest_runs,
    )
    latest_runs = pick_latest_version(latest_runs)
    datasets = []
    for run in sorted(latest_runs):
        print(f"Processing {run}")
        with fs.open(run) as f:
            data = json.load(f)
        try:
            datasets += parse_cmip6_downscaled_pyramid(data, cdn, root_path)

        except Exception as e:
            print(f"Error parsing {run}: {e}")
            continue
    return datasets


@task(log_stdout=True)
def create_catalog(
    *,
    catalog_path: str,
    cmip6_raw_pyramids: list[dict] = None,
    cmip6_downscaled_pyramids: list[dict] = None,
    era5_pyramids: list[dict] = None,
):

    """
    Create catalog.

    Parameters
    ----------
    catalog_path : str
        Path to the catalog.
    cmip6_raw_pyramids : list[dict], optional
        List of CMIP6 raw pyramids. The default is None.
    cmip6_downscaled_pyramids : list[dict], optional
        List of CMIP6 downscaled pyramids. The default is None.
    era5_pyramids : list[dict], optional
        List of ERA5 pyramids. The default is None.

    Returns
    -------
    None
    """

    datasets = []
    if cmip6_raw_pyramids:
        datasets += cmip6_raw_pyramids
    if cmip6_downscaled_pyramids:
        datasets += cmip6_downscaled_pyramids
    if era5_pyramids:
        datasets += era5_pyramids

    catalog = {
        "version": "v1.0.0",
        "title": "CMIP6 downscaling catalog",
        "description": "",
        "history": "",
        "last_updated": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": datasets,
    }

    fs = fsspec.filesystem('az', account_name='cmip6downscaling')
    with fs.open(catalog_path, 'w') as f:
        json.dump(catalog, f, indent=2)

    print(f'web-catalog located at: {catalog_path}')
    print(f'{len(datasets)} datasets are included')
    print(f'Sample datasets: {random.sample(datasets, 2)}')


with Flow(
    'web-catalog', executor=runtime.executor, run_config=runtime.run_config, storage=runtime.storage
) as flow:

    paths = Parameter('paths', default=['az://flow-outputs/results/cmip6-pyramids-raw'])
    web_catalog_path = Parameter(
        'web-catalog-path',
        default='az://scratch/results/pyramids/combined-cmip6-era5-pyramids-catalog-web.json',
    )
    downscaled_pyramids_path = Parameter(
        'downscaled-pyramids-path',
        default=r'az://flow-outputs/results/0.1.[^\d+$]*/runs/*/latest.json',
    )

    # https://cmip6downscaling.azureedge.net
    cdn = Parameter('cdn', default='https://cmip6downscaling.blob.core.windows.net')

    root_path = Parameter('root-path', 'https://cmip6downscaling.blob.core.windows.net')

    cmip6_raw_pyramids = get_cmip6_pyramids(paths, cdn, root_path)
    minimum_version = Parameter('minimum-version', default='0.1.8')
    maximum_version = Parameter('maximum-version', default=None)
    exclude_local_version = Parameter('exclude-local-version', default=False)

    cmip6_downscaled_pyramids = get_cmip6_downscaled_pyramids(
        path=downscaled_pyramids_path,
        cdn=cdn,
        root_path=root_path,
        minimum_version=minimum_version,
        maximum_version=maximum_version,
        exclude_local_version=exclude_local_version,
    )
    create_catalog(
        catalog_path=web_catalog_path,
        cmip6_raw_pyramids=cmip6_raw_pyramids,
        cmip6_downscaled_pyramids=cmip6_downscaled_pyramids,
    )
