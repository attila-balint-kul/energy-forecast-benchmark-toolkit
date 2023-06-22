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

    @app.post("/predict")
    async def predict(
        horizon: int,
        y: Annotated[bytes, File()],
        # X: Annotated[bytes, File()],
        level: Optional[List[int]] = Query(None),
    ):
        y_df = pd.read_parquet(io.BytesIO(y))
        # X_df = pd.read_parquet(io.BytesIO(X))

        y_df["ds"] = pd.to_datetime(y_df["ds"])
        y = y_df.set_index("ds").y

        forecast = model.predict(
            h=horizon,
            y=y,
            # X=X_df,
            level=level,
        )
        forecast.fillna(0, inplace=True)

        response = {
            "forecast": jsonable_encoder(forecast.to_dict(orient="records")),
        }
        return JSONResponse(
            content=response,
            status_code=200,
        )

    return app
