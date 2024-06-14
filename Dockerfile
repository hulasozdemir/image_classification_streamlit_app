FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
       build-essential \
       libglib2.0-0 \
       libsm6 \
       libxext6 \
       libxrender-dev \
       pkg-config \
       libhdf5-dev \
       && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py"]
