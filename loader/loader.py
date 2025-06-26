import os
import uuid
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pathlib import Path
import time
import streamlit as st
import zipfile
import tempfile
import shutil
import requests
import json
from config import EMBEDDING_SERVICE_URL, COLLECTION_NAME, ROW_BATCH_SIZE, EMBEDDING_BATCH_SIZE, QDRANT_HOST, QDRANT_PORT


# Инициализация клиента Qdrant (кэширование в Streamlit)
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def get_embedding_service_info():
    """Получение информации о сервисе эмбеддингов"""
    try:
        response = requests.get(f"{EMBEDDING_SERVICE_URL}/model-info")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(
                f"Ошибка получения информации о модели: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Не удается подключиться к сервису эмбеддингов: {e}")
        return None


def generate_embeddings_via_service(texts):
    """Генерация эмбеддингов через внешний сервис"""
    try:
        payload = {
            "texts": texts,
            "task": "retrieval.passage"
        }

        response = requests.post(
            f"{EMBEDDING_SERVICE_URL}/embed",
            json=payload,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            return result["embeddings"]
        else:
            st.error(f"Ошибка генерации эмбеддингов: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Ошибка подключения к сервису эмбеддингов: {e}")
        return None


def extract_zip_to_temp(zip_file):
    """Извлечение ZIP-архива во временную папку"""
    temp_dir = tempfile.mkdtemp()

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    return temp_dir


def load_markdown_content(file_path, base_path):
    """Загрузка содержимого markdown файла"""
    try:
        full_path = Path(base_path) / file_path
        with open(full_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        st.error(f"Ошибка при чтении файла {file_path}: {e}")
        return ""


def process_csv_batch(csv_rows, base_path):
    """Обработка батча строк CSV с созданием двух векторов на объект"""
    object_name_texts = []
    friendly_name_texts = []
    metadatas = []

    for _, row in csv_rows.iterrows():
        object_name = row["Имя объекта"]
        object_type = row["Тип объекта"]
        synonym = row["Синоним"]
        file_name = row["Файл"]

        # Загружаем содержимое markdown файла
        doc = load_markdown_content(file_name, base_path)
        metadata = {
            "object_name": object_name,
            "object_type": object_type,
            "doc": doc,
            "file_name": file_name
        }

        # Создаем два типа текстов для векторизации:
        # 1. object_name - точное полное имя объекта как в терминах 1С
        object_name_text = object_name

        # 2. friendly_name - более удобное для чтения имя
        friendly_name_text = f"{object_type}: {synonym}"

        object_name_texts.append(object_name_text)
        friendly_name_texts.append(friendly_name_text)
        metadatas.append(metadata)

    return object_name_texts, friendly_name_texts, metadatas


def generate_embeddings_batch(texts, batch_info_text, embedding_progress_bar):
    """Генерация эмбеддингов батчами через внешний сервис"""
    all_embeddings = []

    total_embedding_batches = (
        len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch_texts = texts[i:i + EMBEDDING_BATCH_SIZE]
        current_batch = i // EMBEDDING_BATCH_SIZE + 1

        batch_info_text.write(
            f"Обработано {current_batch} батчей ембеддингов из {total_embedding_batches}")
        embedding_progress_bar.progress(
            current_batch / total_embedding_batches)

        # Генерируем эмбеддинги через внешний сервис
        batch_embeddings = generate_embeddings_via_service(batch_texts)
        if batch_embeddings is None:
            st.error("Ошибка генерации эмбеддингов")
            return None

        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def upload_to_qdrant(object_name_embeddings, friendly_name_embeddings, object_name_texts, friendly_name_texts, metadatas, client, collection_name):
    """Загрузка в Qdrant с двумя типами векторов"""
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "object_name": object_name_embedding,
                "friendly_name": friendly_name_embedding
            },
            payload={
                # "object_name_text": object_name_text,
                # "friendly_name_text": friendly_name_text,
                "friendly_name": friendly_name_text,
                **metadata
            }
        )
        for object_name_embedding, friendly_name_embedding, object_name_text, friendly_name_text, metadata in
        zip(object_name_embeddings, friendly_name_embeddings,
            object_name_texts, friendly_name_texts, metadatas)
    ]

    client.upsert(
        collection_name=collection_name,
        points=points
    )


def process_files(zip_file, collection_name):
    """Основная функция обработки файлов"""
    temp_dir = None
    try:
        # Извлечение ZIP-архива
        st.write("Извлечение ZIP-архива...")
        temp_dir = extract_zip_to_temp(zip_file)

        # Поиск файла objects.csv в извлеченной папке
        csv_path = None
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.lower() == 'objects.csv':
                    csv_path = os.path.join(root, file)
                    break
            if csv_path:
                break

        if not csv_path:
            st.error("Файл objects.csv не найден в архиве")
            return False

        # Инициализация клиентов
        client = get_qdrant_client()

        # Проверяем доступность сервиса эмбеддингов
        st.write("Проверка подключения к сервису эмбеддингов...")
        embedding_info = get_embedding_service_info()
        if not embedding_info:
            st.error(
                "Не удается подключиться к сервису эмбеддингов. Убедитесь, что сервис запущен на http://localhost:5000")
            return False

        st.write(
            f"Подключен к сервису эмбеддингов. Модель: {embedding_info.get('model_name', 'unknown')}, размерность: {embedding_info.get('dimensions', 'unknown')}")

        # Обновляем размерность на основе информации от сервиса
        global DIMENSIONS
        DIMENSIONS = embedding_info.get('dimensions', 384)

        # Проверка и пересоздание коллекции
        if client.collection_exists(collection_name):
            collection_info = client.get_collection(collection_name)
            if collection_info.points_count > 0:
                st.write(
                    f"Удаление существующей коллекции {collection_name} с {collection_info.points_count} записями...")
                client.delete_collection(collection_name)
                st.write("Коллекция удалена.")

        if not client.collection_exists(collection_name):
            st.write(
                f"Создание новой коллекции {collection_name} с поддержкой двух типов векторов...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "object_name": VectorParams(
                        size=DIMENSIONS,
                        distance=Distance.COSINE,
                        on_disk=True
                    ),
                    "friendly_name": VectorParams(
                        size=DIMENSIONS,
                        distance=Distance.COSINE,
                        on_disk=True
                    )
                }
            )
            st.write("Коллекция создана.")

        # Чтение CSV файла
        try:
            df = pd.read_csv(csv_path, encoding='utf-8',
                             sep=';', quotechar='"')
            st.write(f"Загружен CSV файл с {len(df)} строками")
            # st.write(f"Колонки: {list(df.columns)}")

            # Проверяем наличие необходимых колонок
            required_columns = ["Имя объекта",
                                "Тип объекта", "Синоним", "Файл"]
            missing_columns = [
                col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Отсутствуют колонки: {missing_columns}")
                return False

        except Exception as e:
            st.error(f"Ошибка при чтении CSV файла: {e}")
            return False

        total_rows = len(df)
        total_points_processed = 0

        # Создаем общий прогресс-бар для обработки строк CSV
        csv_progress_text = st.empty()
        overall_progress = st.progress(0)

        # Создаем прогресс-бар для генерации эмбеддингов
        batch_info_text = st.empty()
        embedding_progress = st.progress(0)

        # Обработка строк CSV батчами
        for i in range(0, total_rows, ROW_BATCH_SIZE):
            row_batch = df.iloc[i:i + ROW_BATCH_SIZE]
            current_batch_num = i // ROW_BATCH_SIZE + 1
            total_batches = (total_rows + ROW_BATCH_SIZE - 1) // ROW_BATCH_SIZE

            # Обновляем общий прогресс
            rows_processed = min(i + ROW_BATCH_SIZE, total_rows)
            csv_progress_text.write(
                f"Обработка строк CSV: {rows_processed}/{total_rows} (батч строк {current_batch_num}/{total_batches})")
            overall_progress.progress(rows_processed / total_rows)

            # Обработка текста
            object_name_texts, friendly_name_texts, metadatas = process_csv_batch(
                row_batch, temp_dir)

            if not object_name_texts:
                st.warning("Нет текстов для обработки в этом батче")
                continue

            # Генерация эмбеддингов для object_name
            object_name_embeddings = generate_embeddings_batch(
                object_name_texts, batch_info_text, embedding_progress)

            if object_name_embeddings is None:
                st.error(
                    "Ошибка генерации эмбеддингов для object_name, прерываем обработку")
                return False

            # Генерация эмбеддингов для friendly_name
            friendly_name_embeddings = generate_embeddings_batch(
                friendly_name_texts, batch_info_text, embedding_progress)

            if friendly_name_embeddings is None:
                st.error(
                    "Ошибка генерации эмбеддингов для friendly_name, прерываем обработку")
                return False

            # Загрузка в Qdrant
            upload_to_qdrant(object_name_embeddings, friendly_name_embeddings,
                             object_name_texts, friendly_name_texts, metadatas,
                             client, collection_name)

            total_points_processed += len(object_name_texts)

            time.sleep(1)

        st.success(
            f"Обработка завершена! Всего загружено {total_points_processed} записей в коллекцию {collection_name}")
        return True

    except Exception as e:
        st.error(f"Ошибка при обработке файлов: {e}")
        return False

    finally:
        # Очистка временных файлов
        if temp_dir and os.path.exists(temp_dir):
            st.write("Очистка временных файлов...")
            shutil.rmtree(temp_dir)
            st.write("Временные файлы удалены.")


def main():
    st.title("Загрузка данных в Qdrant")
    st.write("Загрузите ZIP-архив с markdown файлами и файлом objects.csv")

    # Поле для ввода имени коллекции
    collection_name = st.text_input(
        "Имя коллекции Qdrant",
        value=COLLECTION_NAME,
        help="Введите имя коллекции, в которую будут загружены данные"
    )

    # Проверяем, что имя коллекции не пустое
    if not collection_name.strip():
        st.error("Пожалуйста, введите имя коллекции")
        collection_name = COLLECTION_NAME

    # Загрузка файлов
    zip_file = st.file_uploader(
        "Выберите ZIP архив",
        type=['zip'],
        help="Архив должен содержать markdown файлы и файл objects.csv"
    )

    # Кнопка запуска обработки
    if st.button("Начать обработку", type="primary"):
        if zip_file is None:
            st.error("Пожалуйста, загрузите ZIP архив")
        elif not collection_name.strip():
            st.error("Пожалуйста, введите имя коллекции")
        else:
            with st.spinner("Обработка файлов..."):
                success = process_files(zip_file, collection_name.strip())
                if success:
                    st.balloons()


if __name__ == "__main__":
    main()
