---
hide:
  - navigation
---
# Examples

This repository contains example models and notebooks to get started with the benchmark toolkit.
The examples models are found in the `models/` folder, and the example notebooks are in the `notebooks/` folder.

## Folder Structure

The repository follows this structure:

```
├── README.md                       <- The top-level README for getting started.
├── data
│   ├── demand.parquet              <- Demand data subset.
│   ├── metadata.parquet            <- Metadata subset.
│   └── weather.parquet             <- Weather data subset.
│
├── models                          <- Example models each in its own subfolder.
│   ├── sf-naive-seasonal           <- Naive seasonal model based on statsforecast package.
│   │   ├── src                     <- Source code for the model.
│   │   │   └── main.py             <- Entrypoint for the forecast server.
│   ├── Dockerfile                  <- Example Dockerfile for the model. 
│   └── requirements.txt            <- Model's requirements.
│
├── notebooks                       <- Jupyter notebooks, should be read in order.
│   ├── 01. Univariate.ipynb        <- Simple univariate forecast model benchmarking example.
│   ├── 02. Multivariate.ipynb      <- Multivariate forecast model benchmarking example.
│   └── 02. ForecastClient.ipynb    <- Benchmarking using the ForecastClient example.
│
└── requirements.txt                <- Overall requirements to run all the example notebooks.
```

## Requirements

To contribute models to the benchmark, you need to have Docker installed. 
Follow the installation procedure for your platform on the [docker website](https://www.docker.com/products/docker-desktop/).

## Getting Started

Clone this repository:
```bash
git clone https://github.com/attila-balint-kul/energy-forecast-benchmark-toolkit
cd energy-forecast-benchmark-toolkit
```

Install the requirements (recommended inside a virtual environment):
```bash
pip install notebook enfobench
```

To run the notebooks, you also need the HuggingFace dataset [attila-balint-kul/electricity-demand](https://huggingface.co/datasets/attila-balint-kul/electricity-demand).
Download all three files from the `data/` folder into the `data/` folder of this repository.

Run the example notebooks in the `notebooks` folder.

## Creating a Model

To create a model, use the `models/sf-naive/` folder as a template. 
If you follow the folder structure, have a `requirements.txt` file, 
and all your source code is inside the `src/` folder, there is generally 
no need to change the `Dockerfile`.
Once your model is ready, you can build the docker image:

```bash
docker build -t tag-that-identifies-the-model ./path/to/the/folder/containing/the/Dockerfile
```

To run the Docker image:
```bash
docker run -p 3000:3000 tag-that-identifies-the-model
```

Then you can test your model by using the `03. ForecastClient.ipynb` notebook.

Once the model is tested, push it to any public Docker registry 
(e.g., DockerHub). Contact us with the repository and model tag, 
and we will add it to the [dashboard](https://wandb.ai/attila-balint-kul/load-forecasting-competition/reports/Enfobench-Dashboard--Vmlldzo2MDM0ODE2#models).
