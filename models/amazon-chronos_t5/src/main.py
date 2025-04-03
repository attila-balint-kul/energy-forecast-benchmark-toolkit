import os
from pathlib import Path

import pandas as pd
import torch
from chronos import ChronosPipeline

from enfobench import AuthorInfo, ForecasterType, ModelInfo
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index, periods_in_duration


root_dir = Path(__file__).parent.parent
# Check for GPU availability
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class AmazonChronosT5Model:
    def __init__(self, model_name: str, num_samples: int, ctx_length: str | None = None):
        self.model_name = model_name
        self.num_samples = num_samples
        self.ctx_length = ctx_length

    def info(self) -> ModelInfo:
        name = (
            "Amazon-"
            f'{"-".join(map(str.capitalize, self.model_name.split("-")))}'
            f'{"-CTX" + self.ctx_length if self.ctx_length else ""}'
        )
        return ModelInfo(
            name=name,
            authors=[
                AuthorInfo(name="Attila Balint", email="attila.balint@kuleuven.be"),
            ],
            type=ForecasterType.quantile,
            params={
                "num_samples": self.num_samples,
                "ctx_length": self.ctx_length,
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
            msg = f"Model directory for {self.model_name} was not found at {model_dir}, make sure it is downloaded."
            raise FileNotFoundError(msg)

        pipeline = ChronosPipeline.from_pretrained(
            model_dir,
            device_map=device,
            torch_dtype=torch.float32,
        )

        # context must be either a 1D tensor, a list of 1D tensors,
        # or a left-padded 2D tensor with batch as the first dimension
        if self.ctx_length is None:
            context = torch.tensor(history.y)
        else:
            ctx_length = min(periods_in_duration(history.index, duration=self.ctx_length), len(history))
            context = torch.tensor(history.y[-ctx_length:])

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


model_name = os.getenv("ENFOBENCH_MODEL_NAME")
num_samples = int(os.getenv("ENFOBENCH_NUM_SAMPLES"))
ctx_length = os.getenv("ENFOBENCH_CTX_LENGTH")

# Instantiate your model
model = AmazonChronosT5Model(model_name=model_name, num_samples=num_samples, ctx_length=ctx_length)

# Create a forecast server by passing in your model
app = server_factory(model)
