import concurrent.futures
import csv
import io
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import xarray as xr
from obstore.auth.planetary_computer import PlanetaryComputerCredentialProvider
from obstore.store import AzureStore, HTTPStore
from zarr.storage import ObjectStore

MSFT_DATA = "https://cpdataeuwest.blob.core.windows.net/cp-cmip/version1/data"
OSN_DATA = "https://rice1.osn.mghpcc.org/carbonplan/cp-cmip/version1/data"
OSN_RECHUNKED = "https://rice1.osn.mghpcc.org/carbonplan/cp-cmip/version1/rechunked_data"
CATALOG_CMD = [
    "rclone",
    "cat",
    "osncarbonplan:carbonplan/cp-cmip/version1/catalog/osn-global-downscaled-cmip6.csv",
]
OUT_PATH = Path("audit_results.csv")


def sample_value(store: ObjectStore) -> float | None:
    try:
        ds = xr.open_zarr(store, chunks=None, zarr_format=2)
        var = list(ds.data_vars)[0]
        val = float(ds[var].isel(time=0, lat=len(ds.lat) // 2, lon=len(ds.lon) // 2).values)
        ds.close()
        return val
    except Exception:
        return None


def is_valid(val: float | None) -> bool:
    return val is not None and not np.isnan(val)


def classify(msft_val: float | None, osn_data_val: float | None, osn_rechunked_val: float | None) -> str:
    if not is_valid(msft_val) and not is_valid(osn_data_val):
        return "bad_original_write"
    if is_valid(msft_val) and not is_valid(osn_data_val):
        return "bad_osn_copy"
    if is_valid(osn_data_val) and not is_valid(osn_rechunked_val):
        return "bad_rechunk"
    if not is_valid(msft_val) and is_valid(osn_data_val):
        return "osn_fixed_somehow"
    return "ok"


def audit_store(row: dict) -> dict:
    store_name = Path(row["downscaled_daily_data_uri"]).name
    method = row["method"]

    msft_store = ObjectStore(AzureStore(credential_provider=PlanetaryComputerCredentialProvider(
        f"{MSFT_DATA}/{method}/{store_name}"
    )))
    osn_data_store = ObjectStore(HTTPStore(f"{OSN_DATA}/{method}/{store_name}"))
    osn_rechunked_store = ObjectStore(HTTPStore(f"{OSN_RECHUNKED}/{method}/{store_name}"))

    msft_val = sample_value(msft_store)
    osn_data_val = sample_value(osn_data_store)
    osn_rechunked_val = sample_value(osn_rechunked_store)
    cls = classify(msft_val, osn_data_val, osn_rechunked_val)

    print(
        f"{cls:>22}  msft={str(msft_val):>8}  osn_data={str(osn_data_val):>8}"
        f"  osn_rechunked={str(osn_rechunked_val):>8}  {method}/{store_name}"
    )
    return {
        "method": method,
        "store": store_name,
        "variable": row["variable_id"],
        "source_id": row["source_id"],
        "experiment_id": row["experiment_id"],
        "msft_val": msft_val,
        "osn_data_val": osn_data_val,
        "osn_rechunked_val": osn_rechunked_val,
        "classification": cls,
    }


def main() -> None:
    raw = subprocess.check_output(CATALOG_CMD)
    rows = [r for r in csv.DictReader(io.StringIO(raw.decode())) if r["timescale"] == "day"]
    print(f"Auditing {len(rows)} day stores across MSFT and OSN...\n")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(audit_store, r): r for r in rows}
        for fut in concurrent.futures.as_completed(futs):
            results.append(fut.result())

    fieldnames = ["method", "store", "variable", "source_id", "experiment_id",
                  "msft_val", "osn_data_val", "osn_rechunked_val", "classification"]
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: (r["method"], r["store"])))

    counts = Counter(r["classification"] for r in results)
    print("\n--- Summary ---")
    for cls, n in sorted(counts.items()):
        print(f"  {cls}: {n}")
    print(f"\nFull results: {OUT_PATH}")


if __name__ == "__main__":
    main()
