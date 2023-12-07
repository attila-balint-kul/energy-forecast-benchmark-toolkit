FROM python:3.11.6-slim-bookworm

WORKDIR /usr/local/app

COPY ./requirements.txt /usr/local/app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r /usr/local/app/requirements.txt

# Polars requires AVX2 CPU instructions, which are not available on the server
RUN pip uninstall -y polars
RUN pip install --no-cache-dir polars-lts-cpu

COPY ./src /usr/local/app/src

ENV ENFOBENCH_MODEL_SEASONALITY="1D"

EXPOSE 3000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "3000"]
