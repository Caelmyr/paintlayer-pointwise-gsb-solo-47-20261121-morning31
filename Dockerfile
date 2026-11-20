ARG PAINTLAYER_DEPS_IMAGE=python:3.11-slim
FROM ${PAINTLAYER_DEPS_IMAGE} AS dependencies
USER root
WORKDIR /build/paintlayer
COPY requirements.txt .
RUN python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: -r requirements.txt && python -m pip check

FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HOST=0.0.0.0 PORT=8000 PAINTLAYER_DATA=/var/lib/paintlayer
WORKDIR /opt/paintlayer
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY paintlayer ./paintlayer
COPY static ./static
RUN useradd -U -u 11500 paintlayer && install -d -o paintlayer -g paintlayer /var/lib/paintlayer
USER paintlayer
EXPOSE 8000
CMD ["python", "-m", "paintlayer"]
