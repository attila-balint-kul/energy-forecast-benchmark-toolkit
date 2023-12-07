FROM attilabalint/enfobench-demand-models:base-u8darts-0.27.0

WORKDIR /usr/local/app

COPY ./requirements.txt /usr/local/app/requirements.txt
RUN pip install --no-cache-dir -r /usr/local/app/requirements.txt

COPY ./src /usr/local/app/src

ENV ENFOBENCH_MODEL_SEASONALITY="1D"
ENV ENFOBENCH_MODEL_DIRECT="0"

EXPOSE 3000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "3000"]
