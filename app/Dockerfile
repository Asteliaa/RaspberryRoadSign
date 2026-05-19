FROM python:3.10-slim

# Системные зависимости для стабильной работы OpenCV с видеопотоком
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Установка легковесного рантайма без PyTorch
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем начинку папки app в контейнер
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]