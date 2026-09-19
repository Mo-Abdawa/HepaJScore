FROM python:3.12-slim

# System libraries for Pillow + OpenCV-headless + MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
        libjpeg-dev zlib1g-dev libwebp-dev \
        libglib2.0-0 libgl1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/uploads /app/data
VOLUME ["/app/uploads", "/app/data"]

ENV PORT=8000 HOST=0.0.0.0 PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["gunicorn", "-w", "2", "--threads", "4", "-b", "0.0.0.0:8000", \
     "--access-logfile", "-", "app:app"]
