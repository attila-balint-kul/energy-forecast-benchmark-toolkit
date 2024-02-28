FROM attilabalint/enfobench-demand-models:base-sktime-0.26.1

WORKDIR /usr/local/app

COPY ./requirements.txt /usr/local/app/requirements.txt
RUN pip install --no-cache-dir -r /usr/local/app/requirements.txt

COPY ./src /usr/local/app/src

COPY ./src /usr/local/app/src

EXPOSE 3000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "3000"]
