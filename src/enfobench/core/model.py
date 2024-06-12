from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import pandas as pd


class ForecasterType(str, Enum):
    point = "point"
    quantile = "quantile"
    density = "density"
    ensemble = "ensemble"


@dataclass
class AuthorInfo:
    """Author information.

    Attributes:
        name: Name of the author.
        email: Email of the author.
    """

    name: str
    email: str | None = None


@dataclass
class ModelInfo:
    """Model information.

    Attributes:
        name: Name of the model.
        authors: List of authors.
        type: Type of the model.
        params: Parameters of the model.
    """

    name: str
    authors: list[AuthorInfo]
    type: ForecasterType
    params: dict[str, Any] = field(default_factory=dict)


class Model(Protocol):
    def info(self) -> ModelInfo: ...

    def forecast(
        self,
        horizon: int,
        history: pd.DataFrame,
        past_covariates: pd.DataFrame | None = None,
        future_covariates: pd.DataFrame | None = None,
        metadata: dict | None = None,
        level: list[int] | None = None,
        **kwargs,
    ) -> pd.DataFrame: ...
