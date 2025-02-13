FROM python:3.12-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make libffi-dev libssl-dev && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip

COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt --no-cache-dir --verbose

# Install Playwright
RUN playwright install --with-deps

COPY /nltk_data /nltk_data
COPY . .

EXPOSE 5000

ENV NLTK_DATA=/nltk_data
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:create_app()"]
