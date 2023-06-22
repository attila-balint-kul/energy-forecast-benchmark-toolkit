from enum import Enum
from typing import Any, Protocol, Optional, List

import pandas as pd
from pydantic import BaseModel


class ForecasterType(str, Enum):
    point = "point"
    quantile = "quantile"
    density = "density"
    ensemble = "ensemble"


class ModelInfo(BaseModel):
    name: str
    type: ForecasterType
    params: dict[str, Any]


class EnvironmentInfo(BaseModel):
    packages: dict[str, str]


class Model(Protocol):
    def info(self) -> ModelInfo:
        ...

    def predict(
        self,
        h: int,
        y: pd.Series,
        X: Optional[pd.DataFrame] = None,
        level: Optional[List[int]] = None,
        **kwargs,
    ) -> pd.DataFrame:
        ...
