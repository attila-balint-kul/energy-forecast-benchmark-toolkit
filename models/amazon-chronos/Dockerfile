FROM attilabalint/enfobench-models:base-amazon-chronos-1.2.0

WORKDIR /usr/local/app

COPY ./requirements.txt /usr/local/app/requirements.txt
RUN pip install --no-cache-dir -r /usr/local/app/requirements.txt

ARG MODEL_SIZE
COPY ./models/chronos-t5-${MODEL_SIZE} /usr/local/app/models/chronos-t5-${MODEL_SIZE}
COPY ./src /usr/local/app/src

ENV ENFOBENCH_MODEL_NAME="chronos-t5-${MODEL_SIZE}"
ENV ENFOBENCH_NUM_SAMPLES="1"

EXPOSE 3000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "1"]
