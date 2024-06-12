import io
import json
import sys
from typing import Annotated, Any

import pandas as pd
import pkg_resources
from fastapi import FastAPI, File, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.responses import RedirectResponse

from enfobench.core import ForecasterType, Model


class AuthorInfo(BaseModel):
    """Author information.

    Attributes:
        name: Name of the author.
        email: Email of the author.
    """

    name: str
    email: str | None = None


class ModelInfo(BaseModel):
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
    params: dict[str, Any]


class EnvironmentInfo(BaseModel):
    python: str
    packages: dict[str, str]


def server_factory(model: Model) -> FastAPI:
    app = FastAPI()
    environment = EnvironmentInfo(
        python=sys.version,
        packages={package.key: package.version for package in pkg_resources.working_set},
    )

    @app.get("/", include_in_schema=False)
    async def index():
        return RedirectResponse(url="/docs")

    @app.get("/info", response_model=ModelInfo)
    async def model_info():
        """Return model information."""
        return model.info()

    @app.get("/environment", response_model=EnvironmentInfo)
    async def environment_info():
        """Return information of installed packages and their versions."""
        return environment

    @app.post("/forecast")
    async def forecast(
        horizon: int,
        history: Annotated[bytes, File()],
        past_covariates: Annotated[bytes | None, File()] = None,
        future_covariates: Annotated[bytes | None, File()] = None,
        metadata: Annotated[bytes | None, File()] = None,
        level: list[int] | None = Query(None),  # noqa: B008
    ):
        history_df = pd.read_parquet(io.BytesIO(history))
        past_covariates_df = pd.read_parquet(io.BytesIO(past_covariates)) if past_covariates is not None else None
        future_covariates_df = pd.read_parquet(io.BytesIO(future_covariates)) if future_covariates is not None else None
        metadata = json.load(io.BytesIO(metadata)) if metadata is not None else None

        forecast_df = model.forecast(
            horizon=horizon,
            history=history_df,
            past_covariates=past_covariates_df,
            future_covariates=future_covariates_df,
            metadata=metadata,
            level=level,
        )
        forecast_df.fillna(0, inplace=True)
        forecast_df.rename_axis("timestamp", inplace=True)
        forecast_df.reset_index(inplace=True)

        response = {
            "forecast": jsonable_encoder(forecast_df.to_dict(orient="records")),
        }
        return JSONResponse(
            content=response,
            status_code=200,
        )

    return app
