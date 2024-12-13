FROM python:3.10-slim

# Install graphic libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

WORKDIR /app

COPY requirements.txt .
RUN python3 -m venv venv && \
    ./venv/bin/pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]