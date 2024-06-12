# Energy  Forecast Benchmark Toolkit

[![PyPI version](https://badge.fury.io/py/enfobench.svg)](https://badge.fury.io/py/enfobench)
[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![code style - Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v0.json)](https://github.com/charliermarsh/ruff)
[![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy)

Energy Forecast Benchmark Toolkit is a Python project that aims to provide common tools to
benchmark forecast models.

---

**Documentation**: https://attila-balint-kul.github.io/energy-forecast-benchmark-toolkit/


## Datasets

- **[Electricity demand](https://huggingface.co/datasets/EDS-lab/electricity-demand)**
- **[Gas demand](https://huggingface.co/datasets/EDS-lab/gas-demand)**
- **[PV generation](https://huggingface.co/datasets/EDS-lab/pv-generation)**


## Dashboards

- **[Electricity demand](https://huggingface.co/spaces/EDS-lab/EnFoBench-ElectricityDemand)**
- **[Gas demand](https://huggingface.co/spaces/EDS-lab/EnFoBench-GasDemand)**
- **[PV generation](https://huggingface.co/spaces/EDS-lab/EnFoBench-PVGeneration)**


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

Use the package manager pip to install foobar.

```bash
pip install enfobench
```

## Usage

Download the HuggingFace Dataset ['EDS-lab/electricity-demand'](https://huggingface.co/datasets/EDS-lab/electricity-demand),
and download the files from the data folder to your computer.

```python
import pandas as pd

from enfobench import Dataset
from enfobench.datasets import ElectricityDemandDataset
from enfobench.evaluation import cross_validate, evaluate_metrics
from enfobench.evaluation.metrics import mean_bias_error, mean_absolute_error, root_mean_squared_error

# Load the dataset from the folder that you downloaded the files to.
ds = ElectricityDemandDataset("/path/to/the/dataset/folder/that/contains/all/subsets")

# List all meter ids
ds.list_unique_ids()

# Get one of the meter ids
unique_id = ds.list_unique_ids()[0]

# Get dataset for a specific meter id
target, past_covariates, metadata = ds.get_data_by_unique_id(unique_id)

# Create a dataset
dataset = Dataset(
    target=target,
    past_covariates=past_covariates,
    future_covariates=None,
    metadata=metadata
)

# Import your model and instantiate it
model = MyForecastModel()

# Run cross validation on your model
cv_results = cross_validate(
    model,
    dataset,
    start_date=pd.Timestamp("2018-01-01"),
    end_date=pd.Timestamp("2018-01-31"),
    horizon=pd.Timedelta("24 hours"),
    step=pd.Timedelta("1 day"),
)

# Simply pass in the cross validation results and the metrics you want to evaluate.
metrics = evaluate_metrics(
    cv_results,
    metrics={
        "MBE": mean_bias_error,
        "MAE": mean_absolute_error,
        "RMSE": root_mean_squared_error,
    },
)
```

To get started with some examples check out the `models` folder and the [examples](https://attila-balint-kul.github.io/energy-forecast-benchmark-toolkit/examples) section of the documentation.

## Benchmarking

Once confident in your model, you can submit for evaluation.
The results of the benchmarks are openly accessible through various dashboards. The links you can find above.


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

BSD-3-Clause license
