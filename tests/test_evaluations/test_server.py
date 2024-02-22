from http import HTTPStatus

import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from enfobench.evaluation.client import df_to_buffer
from enfobench.evaluation.server import server_factory


@pytest.fixture(scope="function")
def forecast_client(model):
    app = server_factory(model)
    client = TestClient(app)
    return client


def test_info_endpoint(forecast_client):
    response = forecast_client.get("/info")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "name": "TestModel",
        "authors": [
            {
                "name": "Author 1",
                "email": "author-1@institution.org",
            }
        ],
        "type": "point",
        "params": {
            "param1": 1,
        },
    }


def test_environment_endpoint(forecast_client):
    response = forecast_client.get("/environment")

    assert response.status_code == HTTPStatus.OK
    assert "packages" in response.json()
    for package_name, package_version in response.json()["packages"].items():
        assert isinstance(package_name, str)
        assert isinstance(package_version, str)


def test_forecast_endpoint(forecast_client):
    horizon = 24
    target_index = pd.date_range(start="2020-01-01", end="2021-01-01", freq="1H")
    history_df = pd.DataFrame(
        index=target_index,
        data={
            "y": np.random.normal(size=len(target_index)),
        },
    )

    response = forecast_client.post(
        "/forecast",
        params={
            "horizon": horizon,
        },
        files={
            "history": df_to_buffer(history_df),
        },
    )
    assert response.status_code == HTTPStatus.OK
    assert "forecast" in response.json()
    forecast = response.json()["forecast"]
    assert isinstance(forecast, list) and len(forecast) == horizon
    for forecast_item in forecast:
        assert "timestamp" in forecast_item and pd.Timestamp(forecast_item["timestamp"])
        assert "yhat" in forecast_item and isinstance(
            forecast_item["yhat"],
            float | int,
        )
