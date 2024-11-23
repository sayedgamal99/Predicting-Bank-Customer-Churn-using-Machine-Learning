FROM python:3.11.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/app/

COPY models/ /app/models/

CMD ["python", "app/app.py"]

EXPOSE 5000
