Energy  Forecast Benchmark Toolkit
==============================

[![PyPI version](https://badge.fury.io/py/enfobench.svg)](https://badge.fury.io/py/enfobench)
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![code style - Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v0.json)](https://github.com/charliermarsh/ruff)
[![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy)

Energy Forecast Benchmark Toolkit is a Python project that aims to provide common tools to
benchmark forecast models.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

Use the package manager pip to install foobar.

```bash
pip install enfobench
```

## Usage

Load your own data and create a dataset.

```python
import pandas as pd

from enfobench.dataset import Dataset

# Load your datasets
data = pd.read_csv("../path/to/your/data.csv", parse_dates=['timestamp'], index_col='timestamp')

# Create a target DataFrame that has a pd.DatetimeIndex and a column named 'y'
target = data.loc[:, ['target_column']].rename(columns={'target_column': 'y'})

# Add covariates that can be used as past covariates. This also has to have a pd.DatetimeIndex
past_covariates = data.loc[:, ['covariate_1', 'covariate_2']]

# As sometimes it can be challenging to access historical forecasts to use future covariates, 
# the package also has a helper function to create perfect historical forecasts from the past covariates.
from enfobench.dataset.utils import create_perfect_forecasts_from_covariates

# The example below creates simulated perfect historical forecasts with a horizon of 24 hours and a step of 1 day.
future_covariates = create_perfect_forecasts_from_covariates(
    past_covariates,
    horizon=pd.Timedelta("24 hours"),
    step=pd.Timedelta("1 day"),
)

dataset = Dataset(
    target=data['target_column'],
    past_covariates=past_covariates,
    future_covariates=future_covariates,
)
```

The package integrates with the HuggingFace Dataset ['attila-balint-kul/electricity-demand'](https://huggingface.co/datasets/attila-balint-kul/electricity-demand). 
To use this, just download all the files from the data folder to your computer.

```python
from enfobench.dataset import Dataset, DemandDataset

# Load the dataset from the folder that you downloaded the files to.
ds = DemandDataset("/path/to/the/dataset/folder/that/contains/all/subsets")

# List all meter ids
ds.metadata_subset.list_unique_ids()

# Get dataset for a specific meter id
target, past_covariates, metadata = ds.get_data_by_unique_id("unique_id_of_the_meter")

# Create a dataset
dataset = Dataset(
    target=target,
    past_covariates=past_covariates,
    future_covariates=None,
    metadata=metadata
)
```


You can perform a cross validation on any model locally that adheres to the `enfobench.Model` protocol.

```python
import MyModel
import pandas as pd
from enfobench.evaluation import cross_validate

# Import your model and instantiate it
model = MyModel()

# Run cross validation on your model
cv_results = cross_validate(
    model,
    dataset,
    start_date=pd.Timestamp("2018-01-01"),
    end_date=pd.Timestamp("2018-01-31"),
    horizon=pd.Timedelta("24 hours"),
    step=pd.Timedelta("1 day"),
)
```

You can use the same crossvalidation interface with your model served behind an API. 
To make this simple, both a client and a server are provided.

```python
import pandas as pd
from enfobench.evaluation import cross_validate, ForecastClient

# Import your model and instantiate it
client = ForecastClient(host='localhost', port=3000)

# Run cross validation on your model
cv_results = cross_validate(
    client,
    dataset,
    start_date=pd.Timestamp("2018-01-01"),
    end_date=pd.Timestamp("2018-01-31"),
    horizon=pd.Timedelta("24 hours"),
    step=pd.Timedelta("1 day"),
)
```

The package also collects common metrics used in forecasting.

```python
from enfobench.evaluation import evaluate_metrics_on_forecasts

from enfobench.evaluation.metrics import (
    mean_bias_error, 
    mean_absolute_error, 
    mean_squared_error, 
    root_mean_squared_error,
)

# Simply pass in the cross validation results and the metrics you want to evaluate.
metrics = evaluate_metrics_on_forecasts(
    cv_results,
    metrics={
        "mean_bias_error": mean_bias_error,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "root_mean_squared_error": root_mean_squared_error,
    },
)
```

In order to serve your model behind an API, you can use the built in server factory.

```python
import uvicorn
from enfobench.evaluation.server import server_factory

model = MyModel()

# Create a server that serves your model
server = server_factory(model)
uvicorn.run(server, port=3000)
```

## Contributing

Contributions and feedback are welcome! For major changes, please open an issue first to discuss
what you would like to change.

If you'd like to contribute to the project, please follow these steps:

Fork the repository.
Create a new branch for your feature or bug fix.
Make your changes and commit them.
Push your changes to your forked repository.
Submit a pull request describing your changes.

## License

BSD 2-Clause License
