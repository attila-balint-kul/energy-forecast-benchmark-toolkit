FROM attilabalint/enfobench-models:base-amazon-chronos-1.1.0

WORKDIR /usr/local/app

COPY ./requirements.txt /usr/local/app/requirements.txt
RUN pip install --no-cache-dir -r /usr/local/app/requirements.txt

COPY ./models /usr/local/app/models
COPY ./src /usr/local/app/src

ENV ENFOBENCH_MODEL_NAME="chronos-t5-tiny"
ENV ENFOBENCH_NUM_SAMPLES="20"

EXPOSE 3000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "3000"]
