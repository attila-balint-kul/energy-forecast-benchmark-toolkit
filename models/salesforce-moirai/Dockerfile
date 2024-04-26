FROM attilabalint/enfobench-models:base-salesforce-moirai

WORKDIR /usr/local/app

COPY ./requirements.txt /usr/local/app/requirements.txt
RUN pip install --no-cache-dir -r /usr/local/app/requirements.txt

COPY ./models /usr/local/app/models
COPY ./src /usr/local/app/src

ENV ENFOBENCH_MODEL_NAME="moirai-1.0-R-small"
ENV ENFOBENCH_NUM_SAMPLES="1"

EXPOSE 3000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "3000", "--workers", "1"]
