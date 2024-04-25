import os
from pathlib import Path

import pandas as pd
import torch
from chronos import ChronosPipeline

from enfobench import AuthorInfo, ForecasterType, ModelInfo
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
root_dir = Path(__file__).parent.parent


class AmazonChronosModel:

    def __init__(self, model_name: str, num_samples: int):
        self.model_name = model_name
        self.num_samples = num_samples

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f'Amazon.{".".join(map(str.capitalize, self.model_name.split("-")))}',
            authors=[
                AuthorInfo(name="Attila Balint", email="attila.balint@kuleuven.be"),
            ],
            type=ForecasterType.quantile,
            params={
                "num_samples": self.num_samples,
            },
        )

    def forecast(
        self,
        horizon: int,
        history: pd.DataFrame,
        past_covariates: pd.DataFrame | None = None,
        future_covariates: pd.DataFrame | None = None,
        metadata: dict | None = None,
        level: list[int] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        # Fill missing values
        history = history.fillna(history.y.mean())

        model_dir = root_dir / "models" / self.model_name
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory for {self.model_name} was not found at {model_dir}, make sure it is downloaded."
            )
        pipeline = ChronosPipeline.from_pretrained(
            model_dir,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )

        # context must be either a 1D tensor, a list of 1D tensors,
        # or a left-padded 2D tensor with batch as the first dimension
        context = torch.tensor(history.y)
        prediction_length = horizon
        forecasts = pipeline.predict(
            context,
            prediction_length,
            num_samples=self.num_samples,
            limit_prediction_length=False,
        )  # forecast shape: [num_series, num_samples, prediction_length]
        data = {"yhat": forecasts.mean(dim=1)[0]}
        # for lvl in level:
        #     data[f"q{lvl}"] = forecasts.quantile(lvl / 100, dim=1)[0]  # TODO: extend to quantiles

        # Postprocess forecast
        index = create_forecast_index(history=history, horizon=horizon)
        forecast = pd.DataFrame(index=index, data=data)
        return forecast


model_name = os.getenv("MODEL_NAME")
num_samples = int(os.getenv("NUM_SAMPLES"))

# Instantiate your model
model = AmazonChronosModel(model_name=model_name, num_samples=num_samples)

# Create a forecast server by passing in your model
app = server_factory(model)
