FROM python:3.11.6-slim-bookworm

WORKDIR /usr/local/app

ARG DARTS_VERSION=0.27.0

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir u8darts==$DARTS_VERSION

# Polars requires AVX2 CPU instructions, which are not available on the server
RUN pip uninstall -y polars
RUN pip install --no-cache-dir polars-lts-cpu
