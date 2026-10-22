# syntax=docker/dockerfile:1.7
# Fall-detection app. It WATCHES (decodes each camera live at ~10 fps), so
# it needs ffmpeg, and it encodes frames and draws the evidence picture,
# so it needs OpenCV. The models are in the adapter, not here.
#
# Build context is the bundle root:
#   docker build -f app/Dockerfile -t opennvr/fall-detection:local .

FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
        "httpx>=0.27,<1.0" \
        "PyYAML>=6.0,<7.0" \
        "websockets>=13.0" \
        "numpy>=1.26,<3.0" \
        "opencv-python-headless>=4.9,<5.0"

COPY app/opennvr-app-sdk /opt/opennvr-app-sdk
RUN pip install --no-cache-dir /opt/opennvr-app-sdk

COPY app/fall_detection.py  ./fall_detection.py
COPY app/fall_rules.py      ./fall_rules.py
COPY app/config.docker.yml  ./config.yml

ENV PYTHONPATH=/app

ENTRYPOINT ["python", "fall_detection.py"]
CMD ["--config", "config.yml"]
