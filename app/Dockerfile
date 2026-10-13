# Fall Detection App — OpenNVR-native monitoring application
#
# Build:
#   docker build -f Dockerfile -t opennvr/fall-detection:1.0.0 .
#
# Run (subscribes to NATS event bus):
#   docker run --rm \
#     --network opennvr_internal \
#     -v $(pwd)/config.yml:/app/config.yml:ro \
#     opennvr/fall-detection:1.0.0 \
#     --config /app/config.yml
 
FROM python:3.11-slim AS runtime
 
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*
 
WORKDIR /app
 
# Install OpenNVR App SDK
COPY app/opennvr-app-sdk /opt/opennvr-app-sdk
RUN pip install --no-cache-dir /opt/opennvr-app-sdk
 
# Copy app code
COPY app/fall_detection.py  ./fall_detection.py
COPY app/alerts.py          ./alerts.py
COPY app/config.example.yml ./config.example.yml
 
ENV PYTHONUNBUFFERED=1
 
ENTRYPOINT ["python", "fall_detection.py"]
CMD ["--config", "config.yml"]