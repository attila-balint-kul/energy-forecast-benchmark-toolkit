FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends git

WORKDIR /usr/local/app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir git+https://github.com/SalesforceAIResearch/uni2ts.git
