FROM python:3.11.6-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends git

WORKDIR /usr/local/app

ARG CHRONOS_VERSION

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir git+https://github.com/amazon-science/chronos-forecasting.git@v$CHRONOS_VERSION
