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

from enfobench.evaluation import Dataset

# Load your dataset and make sure that the timestamp column in named 'ds' and the target values named 'y'
data = pd.read_csv("../path/to/your/data.csv", parse_dates=['timestamp'], index_col='timestamp')
covariates = data.drop(columns=['target_column'])

dataset = Dataset(
    target=data['target_column'],
    covariates=covariates,
)
```

You can perform a cross validation on any model locally that adheres to the `enfobench.Model` protocol.

```python
import MyModel
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

```python
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

The package also collects common metrics for you that you can quickly evaluate on your results.

```python
from enfobench.evaluation import evaluate_metrics_on_forecasts

from enfobench.evaluation.metrics import (
    mean_bias_error, mean_absolute_error, mean_squared_error, root_mean_squared_error,
)

# Merge the cross validation results with the original data
forecasts = cv_results.merge(dataset.target, on="ds", how="left")

metrics = evaluate_metrics_on_forecasts(
    forecasts,
    metrics={
        "mean_bias_error": mean_bias_error,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "root_mean_squared_error": root_mean_squared_error,
    },
)
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

BSD 3-Clause License
