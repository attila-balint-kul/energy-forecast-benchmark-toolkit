import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from enfobench.evaluation.client import to_buffer
from enfobench.evaluation.server import server_factory


@pytest.fixture(scope="function")
def forecast_client(model):
    app = server_factory(model)
    client = TestClient(app)
    return client


def test_info_endpoint(forecast_client):
    response = forecast_client.get("/info")

    assert response.status_code == 200
    assert response.json() == {
        "name": "TestModel",
        "type": "point",
        "params": {
            "param1": 1,
        },
    }


def test_environment_endpoint(forecast_client):
    response = forecast_client.get("/environment")

    assert response.status_code == 200
    assert "packages" in response.json()
    for package_name, package_version in response.json()["packages"].items():
        assert isinstance(package_name, str)
        assert isinstance(package_version, str)


def test_forecast_endpoint(forecast_client):
    horizon = 24
    target_index = pd.date_range(start="2020-01-01", end="2021-01-01", freq="1H")
    history_df = pd.DataFrame(
        data={
            "y": np.random.normal(size=len(target_index)),
            "ds": target_index,
        }
    )

    response = forecast_client.post(
        "/forecast",
        params={
            "horizon": horizon,
        },
        files={
            "history": to_buffer(history_df),
        },
    )
    assert response.status_code == 200
    assert "forecast" in response.json()
    forecast = response.json()["forecast"]
    assert isinstance(forecast, list) and len(forecast) == horizon
    for forecast_item in forecast:
        assert "ds" in forecast_item and pd.Timestamp(forecast_item["ds"])
        assert "yhat" in forecast_item and isinstance(
            forecast_item["yhat"],
            (
                float,
                int,
            ),
        )
