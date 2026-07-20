FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    libjpeg-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py
COPY rep_sdt2v_ai_ads_video_pipeline.py /app/rep_sdt2v_ai_ads_video_pipeline.py
COPY r2_utils.py /app/r2_utils.py

CMD ["python", "-u", "/app/handler.py"]
