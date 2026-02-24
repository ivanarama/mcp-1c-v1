FROM python:3.11-slim

# Установка системных зависимостей (минимум для streamlit)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование requirements.txt и установка зависимостей
COPY requirements.txt .

# Pip optimizations
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_PYTHON_VERSION_WARNING=1 \
    PIP_PREFER_BINARY=1
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Копирование исходного кода
COPY . .

# Экспонирование порта Streamlit
EXPOSE 8501

# Команда запуска
CMD ["streamlit", "run", "loader.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.maxUploadSize=1000"]
