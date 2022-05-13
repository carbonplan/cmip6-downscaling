from __future__ import annotations

from dataclasses import dataclass

from cmip6_downscaling.utils import str_to_hash


@dataclass
class BBox:
    """Lat/lon bounding box"""

    latmin: float = -90.0
    latmax: float = 90.0
    lonmin: float = -180.0
    lonmax: float = 180

    @property
    def lat_slice(self) -> slice:
        return slice(float(self.latmin), float(self.latmax))

    @property
    def lon_slice(self) -> slice:
        return slice(float(self.lonmin), float(self.lonmax))

    def __str__(self) -> str:
        return f"BBox([{self.latmin}, {self.latmax}, {self.lonmin}, {self.lonmax}])"


@dataclass
class TimePeriod:
    """Time period object"""

    start: str = None
    stop: str = None

    @property
    def time_slice(self):
        return slice(self.start, self.stop)

    def __str__(self) -> str:
        return f"TimePeriod([{self.start}, {self.stop}])"


@dataclass
class CMIP6Experiment:

    model: str
    scenario: str
    member: str


@dataclass
class RunParameters:
    """Run parameters object"""

    method: str
    obs: str
    model: str
    member: str
    grid_label: str
    table_id: str
    scenario: str
    variable: str
    features: list
    latmin: float
    latmax: float
    lonmin: float
    lonmax: float
    train_dates: list
    predict_dates: list
    bias_correction_method: str
    bias_correction_kwargs: dict
    model_type: str
    model_params: dict

    @property
    def bbox(self) -> BBox:
        return BBox(
            latmin=self.latmin,
            latmax=self.latmax,
            lonmin=self.lonmin,
            lonmax=self.lonmax,
        )

    @property
    def train_period(self) -> TimePeriod:
        if len(self.train_dates) != 2:
            raise ValueError("expected train_dates to be a list of length 2")
        return TimePeriod(start=self.train_dates[0], stop=self.train_dates[1])

    @property
    def predict_period(self) -> TimePeriod:
        if len(self.predict_dates) != 2:
            raise ValueError("expected predict_dates to be a list of length 2")
        return TimePeriod(start=self.predict_dates[0], stop=self.predict_dates[1])

    @property
    def experiment(self) -> CMIP6Experiment:
        return CMIP6Experiment(model=self.model, scenario=self.scenario, member=self.member)

    @property
    def run_id(self):
        return f"{self.method}_{self.obs}_{self.model}_{self.member}_{self.scenario}_{self.variable}_{self.latmin}_{self.latmax}_{self.lonmin}_{self.lonmax}_{self.train_dates[0]}_{self.train_dates[1]}_{self.predict_dates[0]}_{self.predict_dates[1]}"

    @property
    def run_id_hash(self):
        return str_to_hash(self.run_id)
