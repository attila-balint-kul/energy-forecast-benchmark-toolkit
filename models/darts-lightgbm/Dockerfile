ARG DARTS_VERSION
FROM attilabalint/enfobench-models:base-u8darts-$DARTS_VERSION

# OpenMP is required by LightGBM
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1

WORKDIR /usr/local/app

COPY ./requirements.txt /usr/local/app/requirements.txt
RUN pip install --no-cache-dir -r /usr/local/app/requirements.txt

COPY ./src /usr/local/app/src

EXPOSE 3000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "3000"]
