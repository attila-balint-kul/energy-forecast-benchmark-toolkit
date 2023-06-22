from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

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
        X: pd.DataFrame | None = None,
        level: list[int] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        ...
