FROM attilabalint/enfobench-demand-models:base-statsforecast-1.5.0

WORKDIR /usr/local/app

COPY ./requirements.txt /usr/local/app/requirements.txt
RUN pip install --no-cache-dir -r /usr/local/app/requirements.txt

COPY ./src /usr/local/app/src

ENV ENFOBENCH_MODEL_SEASONALITY="1D"

EXPOSE 3000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "3000"]
