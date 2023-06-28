import io
from typing import Annotated, List, Optional

import pandas as pd
import pkg_resources
from fastapi import FastAPI, File, Query
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from enfobench.evaluation.protocols import EnvironmentInfo, Model, ModelInfo


def server_factory(model: Model) -> FastAPI:
    app = FastAPI()
    environment = EnvironmentInfo(
        packages={package.key: package.version for package in pkg_resources.working_set}
    )

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
        past_covariates: Annotated[Optional[bytes], File()] = None,
        future_covariates: Annotated[Optional[bytes], File()] = None,
        level: Optional[List[int]] = Query(None),
    ):
        history_df = pd.read_parquet(io.BytesIO(history))
        past_covariates_df = (
            pd.read_parquet(io.BytesIO(past_covariates)) if past_covariates is not None else None
        )
        future_covariates_df = (
            pd.read_parquet(io.BytesIO(future_covariates))
            if future_covariates is not None
            else None
        )

        forecast_df = model.forecast(
            horizon=horizon,
            history=history_df,
            past_covariates=past_covariates_df,
            future_covariates=future_covariates_df,
            level=level,
        )
        forecast_df.fillna(0, inplace=True)

        response = {
            "forecast": jsonable_encoder(forecast_df.to_dict(orient="records")),
        }
        return JSONResponse(
            content=response,
            status_code=200,
        )

    return app
