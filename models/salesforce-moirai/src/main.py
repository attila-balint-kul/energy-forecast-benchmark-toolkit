import os
from pathlib import Path

import pandas as pd
import torch
from gluonts.dataset.pandas import PandasDataset

from enfobench import AuthorInfo, ForecasterType, ModelInfo
from enfobench.evaluation.server import server_factory
from enfobench.evaluation.utils import create_forecast_index
from uni2ts.model.moirai import MoiraiForecast


# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
root_dir = Path(__file__).parent.parent


class SalesForceMoraiModel:
    def __init__(self, model_name: str, num_samples: int):
        self.model_name = model_name
        self.num_samples = num_samples
        self.size = model_name.split("-")[-1]

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=f'Salesforce.Moirai-1.0-R.{self.size.capitalize()}',
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

        # Convert into GluonTS dataset
        ds = PandasDataset(dict(history))

        model_dir = root_dir / "models" / self.model_name
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory for {self.model_name} was not found at {model_dir}, make sure it is downloaded."
            )
        # Prepare pre-trained model
        model = MoiraiForecast.load_from_checkpoint(
            checkpoint_path=str(model_dir / 'model.ckpt'),
            prediction_length=horizon,
            context_length=len(history),
            patch_size='auto',
            num_samples=self.num_samples,
            target_dim=1,
            feat_dynamic_real_dim=ds.num_feat_dynamic_real,
            past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
            map_location=device,
        )

        # Make predictions
        predictor = model.create_predictor(batch_size=32)
        forecasts = next(predictor.predict(ds))
        data = {"yhat": forecasts.mean}  # TODO: extend to quantiles

        # Postprocess forecast
        index = create_forecast_index(history=history, horizon=horizon)
        forecast = pd.DataFrame(index=index, data=data)
        return forecast


model_name = os.getenv("ENFOBENCH_MODEL_NAME", "small")
num_samples = int(os.getenv("ENFOBENCH_NUM_SAMPLES", 1))

# Instantiate your model
model = SalesForceMoraiModel(model_name=model_name, num_samples=num_samples)

# Create a forecast server by passing in your model
app = server_factory(model)
