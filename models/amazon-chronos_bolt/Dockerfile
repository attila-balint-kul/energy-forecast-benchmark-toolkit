ARG CHRONOS_VERSION
FROM attilabalint/enfobench-models:base-amazon-chronos-$CHRONOS_VERSION

WORKDIR /usr/local/app

COPY requirements.txt /usr/local/app/requirements.txt
RUN pip install --no-cache-dir -r /usr/local/app/requirements.txt

ARG MODEL_NAME
COPY ./models/chronos-${MODEL_NAME} /usr/local/app/models/chronos-${MODEL_NAME}
COPY src /usr/local/app/src

ENV ENFOBENCH_MODEL_NAME="chronos-${MODEL_NAME}"
ENV ENFOBENCH_NUM_SAMPLES="1"

EXPOSE 3000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "1"]
