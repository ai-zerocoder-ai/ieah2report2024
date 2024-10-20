import json
import logging
import os
from pathlib import Path
import html  # Для экранирования HTML, если выберете этот подход

import openai
from aiogram import Bot, Dispatcher, types
from aiogram.types import ParseMode
from aiogram.utils import executor
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.dispatcher.filters import Command
from aiogram.utils.markdown import escape_md  # Добавлен импорт для экранирования Markdown

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv

from config import TELEGRAM_BOT_TOKEN, OPENAI_API_KEY

# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Проверка наличия необходимых переменных окружения
if not TELEGRAM_BOT_TOKEN:
    logger.error("Отсутствует TELEGRAM_BOT_TOKEN в конфигурации.")
    exit(1)

if not OPENAI_API_KEY:
    logger.error("Отсутствует OPENAI_API_KEY в конфигурации.")
    exit(1)

# Инициализация OpenAI API
openai.api_key = OPENAI_API_KEY

# Инициализация Telegram бота с использованием MarkdownV2
bot = Bot(token=TELEGRAM_BOT_TOKEN, parse_mode=ParseMode.MARKDOWN_V2)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())

# Инициализация эмбеддингов
embedding = OpenAIEmbeddings()

# Путь к векторной базе данных Chroma
persist_directory = 'chroma_db'

# Загрузка существующей векторной базы данных или создание новой
if Path(persist_directory).exists():
    vectordb = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory
    )
    logger.info("Векторная база данных Chroma загружена.")
else:
    # Если база данных не существует, создаем ее
    with open('knowledge_base.json', 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)

    # Предполагаем, что база знаний представлена как список документов
    documents = []
    for entry in knowledge_base:
        if isinstance(entry, dict):
            # Если запись — словарь, извлекаем 'title' и 'content'
            content = entry.get('content', '')
            title = entry.get('title', 'Untitled Document')
            documents.append(f"Title: {title}\nContent: {content}")
        elif isinstance(entry, str):
            # Если запись — строка, добавляем её напрямую
            documents.append(entry)
        else:
            logger.warning(f"Неизвестный формат записи: {entry}")

    # Разделение на чанки с уменьшенным размером
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Уменьшили размер каждого чанка
        chunk_overlap=200,  # Уменьшили перекрытие
        separators=["\n\n", "\n", " ", ""]
    )
    splitted_texts = text_splitter.split_text("\n\n".join(documents))
    logger.info(f"База знаний разделена на {len(splitted_texts)} чанков.")

    # Преобразование чанков в объекты Document
    doc_objects = [Document(page_content=chunk) for chunk in splitted_texts]

    # Создание и сохранение векторной базы данных
    vectordb = Chroma.from_documents(
        documents=doc_objects,
        embedding=embedding,
        persist_directory=persist_directory
    )
    logger.info("Векторная база данных Chroma создана и сохранена.")

# Функция для поиска релевантных документов
def search_documents(query: str, top_k: int = 5) -> list:
    """
    Поиск релевантных документов в векторной базе данных.
    Возвращает список содержимого релевантных документов.
    """
    results = vectordb.similarity_search(query, k=top_k)
    relevant_docs = [doc.page_content for doc in results]
    return relevant_docs

# Функция для генерации ответа на основе релевантных документов
async def generate_response(user_input: str) -> str:
    """
    Генерирует ответ на основе пользовательского ввода и релевантных документов.
    """
    system_prompt = (
        "You are a helpful assistant. Use the provided knowledge base to answer the user's questions."
    )

    # Поиск релевантных документов
    relevant_docs = search_documents(user_input, top_k=5)
    relevant_text = "\n\n".join(relevant_docs)

    user_prompt = (
        f"Вопрос пользователя: {user_input}\n\n"
        f"Релевантная информация из базы знаний:\n{relevant_text}\n\n"
        "Ответ ассистента:"
    )

    try:
        # Обновлённый асинхронный вызов OpenAI API
        response = await openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Убедитесь, что модель указана корректно
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=500,
            temperature=0.2,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        answer = response.choices[0].message["content"].strip()
        logger.info(f"Сгенерированный ответ: {answer}")
        return answer

    except Exception as e:
        logger.error(f"Неизвестная ошибка: {e}")
        raise

# Обработчик команды /start
@dp.message_handler(Command("start"))
async def send_welcome(message: types.Message):
    welcome_text = (
        "👋 Это тестовая демонстрация функционала AI-Ассистента, который использует базу знаний международного энергетического агентства по водородной энергетике и на ее основе выдает ответы по запросам пользователя.\n\n"
        "📌 Используйте команду /iea, чтобы задать вопрос. Например:\n"
        "/iea Какой спрос на водород сейчас?\n\n"
        "Функционал AI-Ассистента и база знаний для работы по генерации ответов устанавливаются по желанию заказчика - обращайтесь в tattoointelligence.tech"
    )
    escaped_welcome_text = escape_md(welcome_text, version=2)  # Экранирование
    await message.reply(escaped_welcome_text, parse_mode=ParseMode.MARKDOWN_V2)

# Обработчик команды /iea
@dp.message_handler(Command("iea"))
async def handle_iea_command(message: types.Message):
    user_input = message.get_args()
    if not user_input:
        await message.reply(
            "Пожалуйста, введите ваш вопрос после команды /iea.\nПример: /iea Какой спрос на водород сейчас?",
            parse_mode=ParseMode.MARKDOWN_V2
        )
        return

    try:
        response = await generate_response(user_input)
        escaped_response = escape_md(response, version=2)  # Экранирование
        await message.answer(escaped_response, parse_mode=ParseMode.MARKDOWN_V2)

    except Exception as e:
        logger.error(f"Неизвестная ошибка при обработке запроса пользователя: {e}")
        await message.reply(
            "Извините, произошла неизвестная ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.",
            parse_mode=ParseMode.MARKDOWN_V2
        )

# Обработчик неизвестных команд
@dp.message_handler(lambda message: message.text.startswith('/') and not any(
    message.text.startswith(f"/{cmd}") for cmd in ["start", "iea"]
))
async def handle_unknown_commands(message: types.Message):
    await message.reply(
        "Извините, я не понимаю эту команду. Используйте /start для начала или /iea для вопросов.",
        parse_mode=ParseMode.MARKDOWN_V2
    )

# Запуск бота
if __name__ == '__main__':
    try:
        logger.info("Бот запускается...")
        executor.start_polling(dp, skip_updates=True)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен.")
