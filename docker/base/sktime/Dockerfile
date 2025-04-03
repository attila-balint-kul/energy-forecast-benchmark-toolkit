FROM python:3.12-slim-bookworm

WORKDIR /usr/local/app

ARG SKTIME_VERSION
ARG AVX2_COMPATIBLE=true

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir sktime==$SKTIME_VERSION

# Polars requires AVX2 CPU instructions, which are not available on the university server
# Conditionally uninstall polars AVX2_COMPATIBLE
RUN if [ "$AVX2_COMPATIBLE" == "false" ]; then \
        pip uninstall -y polars && \
        pip install --no-cache-dir polars-lts-cpu; \
    fi \