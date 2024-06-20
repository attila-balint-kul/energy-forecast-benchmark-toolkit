FROM python:3.11-slim-bookworm

WORKDIR /usr/local/app

ARG SKTIME_VERSION

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir sktime==$SKTIME_VERSION

# Polars requires AVX2 CPU instructions, which are not available on the server
RUN pip uninstall -y polars
RUN pip install --no-cache-dir polars-lts-cpu
