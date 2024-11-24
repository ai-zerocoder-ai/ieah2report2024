# bot.py

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import telebot
from telebot import types
from dotenv import load_dotenv

from config import TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, KNOWLEDGE_BASE_PATH, ADMIN_ID

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings  # Обновлённый импорт
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Устанавливаем общий уровень логирования

# Форматтер для логов
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Обработчик для INFO и выше с ротацией
from logging.handlers import RotatingFileHandler

handler_info = RotatingFileHandler("bot_info.log", maxBytes=5 * 1024 * 1024, backupCount=5)
handler_info.setLevel(logging.INFO)
handler_info.setFormatter(formatter)
logger.addHandler(handler_info)

# Обработчик для ERROR и выше с ротацией
handler_error = RotatingFileHandler("bot_error.log", maxBytes=5 * 1024 * 1024, backupCount=5)
handler_error.setLevel(logging.ERROR)
handler_error.setFormatter(formatter)
logger.addHandler(handler_error)

# Обработчик для вывода логов на консоль
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Инициализация Telegram бота
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Параметры для векторной базы данных
PERSIST_DIRECTORY = 'vector_db'

# Инициализация пула потоков
executor = ThreadPoolExecutor(max_workers=10)


def prepare_vector_db(file_path, persist_directory):
    """
    Загружает и подготавливает векторную базу данных из указанного файла.

    :param file_path: Путь к файлу с базой знаний.
    :param persist_directory: Директория для сохранения векторной базы.
    :return: Объект векторной базы данных или None при ошибке.
    """
    try:
        if not os.path.isfile(file_path):
            logger.error(f"Путь к базе знаний должен быть файлом, но получена директория: '{file_path}'")
            return None

        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()
        if not docs:
            logger.error(f"Не удалось загрузить документ из файла: '{file_path}'")
            return None
        logger.info(f"Загружен документ: {file_path}")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000,
            chunk_overlap=500,
            separators=["\n\n", "\n", "(?<=\. )"]
        )
        splitted_texts = splitter.split_documents(docs)
        logger.info(f"Разделено на {len(splitted_texts)} чанков.")

        embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)  # Обновлённый класс
        vectordb = Chroma.from_documents(
            documents=splitted_texts,
            embedding=embedding,
            persist_directory=persist_directory
        )
        # Удалите вызов vectordb.persist()
        # vectordb.persist()
        logger.info("Векторная база данных создана и сохранена.")
        return vectordb
    except FileNotFoundError:
        logger.error(f"Файл не найден: '{file_path}'")
    except PermissionError:
        logger.error(f"Нет доступа к файлу: '{file_path}'")
    except Exception as e:
        logger.error(f"Ошибка при подготовке векторной базы данных: {e}")
    return None


# Загрузка или создание векторной базы данных
if os.path.exists(PERSIST_DIRECTORY):
    try:
        embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)  # Обновлённый класс
        vectordb = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding
        )
        logger.info(f"Загружена векторная база данных с {vectordb.count()} записей.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке векторной базы данных: {e}")
        vectordb = prepare_vector_db(KNOWLEDGE_BASE_PATH, PERSIST_DIRECTORY)
else:
    vectordb = prepare_vector_db(KNOWLEDGE_BASE_PATH, PERSIST_DIRECTORY)

if vectordb is None:
    logger.error("Векторная база данных не может быть инициализирована. Бот не будет работать корректно.")

# Инициализация модели ChatOpenAI
try:
    model = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0.2,
        openai_api_key=OPENAI_API_KEY
    )
    logger.info("Модель ChatOpenAI инициализирована успешно.")
except Exception as e:
    logger.error(f"Ошибка при инициализации модели ChatOpenAI: {e}")
    model = None

# Обновленный шаблон промпта
prompt_template = PromptTemplate(
    input_variables=["context", "question"],  # Заменено 'query' на 'question'
    template=(
        "Ты — эксперт по водородной энергетике, использующий информацию из базы знаний Международного энергетического агентства (IEA).\n"
        "Используя предоставленный контекст, ответь на следующий вопрос максимально полно и точно.\n\n"
        "Контекст:\n{context}\n\n"
        "Вопрос: {question}\n\n"
        "Ответ:"
    )
)

# Инициализация RetrievalQA цепочки с использованием обновленного промпта
if vectordb and model:
    try:
        retriever = vectordb.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt_template},  # Использование обновленного промпта
            return_source_documents=False
        )
        logger.info("Цепочка RetrievalQA инициализирована успешно с кастомным промптом.")
    except Exception as e:
        logger.error(f"Ошибка при инициализации RetrievalQA цепочки: {e}")
        qa_chain = None
else:
    qa_chain = None

def generate_rag_response(question):
    """
    Генерирует ответ на вопрос пользователя с использованием Retrieval-Augmented Generation (RAG).

    :param question: Вопрос пользователя.
    :return: Ответ на вопрос или сообщение об ошибке.
    """
    if not qa_chain:
        return "Бот не настроен корректно. Пожалуйста, попробуйте позже."
    try:
        response = qa_chain.run(question)  # Используем метод 'run', который возвращает строку
        if not response.strip():
            return "Извините, я не смог найти информацию по вашему запросу."
        return response
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа через RAG: {e}")
        return "Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже."

# Список ID администраторов
ADMIN_IDS = {ADMIN_ID}  # Замените на реальные Telegram ID администратора

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    welcome_text = (
        "👋 Привет! Я AI-Ассистент, использующий базу знаний Международного энергетического агентства по водородной энергетике.\n\n"
        "📌 Используйте команду /iea, чтобы задать вопрос. Например:\n"
        "/iea Каков текущий спрос на водород?\n\n"
        "Функционал AI-Ассистента и база знаний могут быть настроены по желанию заказчика. Для этого обращайтесь на tattoointelligence.tech."
    )
    bot.send_message(message.chat.id, welcome_text, parse_mode='Markdown')

@bot.message_handler(commands=['iea'])
def handle_iea_command(message):
    try:
        user_input = message.text.partition(' ')[2].strip()
        if not user_input:
            bot.reply_to(message,
                         "Пожалуйста, введите запрос после команды /iea. Пример:\n/iea Каков текущий спрос на водород?")
            return

        logger.info(f"Получен запрос от пользователя {message.from_user.id}: {user_input}")

        executor.submit(process_and_reply, message, user_input)
    except Exception as e:
        logger.error(f"Ошибка в обработчике команды /iea: {e}", exc_info=True)
        bot.reply_to(message, "Произошла ошибка при обработке команды. Пожалуйста, попробуйте позже.")

def process_and_reply(message, user_input):
    try:
        response = generate_rag_response(user_input)
        bot.send_message(message.chat.id, response, parse_mode='Markdown')
    except Exception as e:
        logger.exception("Ошибка при обработке команды /iea:")
        bot.send_message(message.chat.id,
                         "Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.")

@bot.message_handler(commands=['reload_kb'])
def reload_knowledge_base_command(message):
    if message.from_user.id not in ADMIN_IDS:
        bot.reply_to(message, "У вас нет прав на выполнение этой команды.")
        logger.warning(f"Пользователь {message.from_user.id} попытался использовать команду /reload_kb.")
        return

    try:
        global vectordb, retriever, qa_chain
        vectordb = prepare_vector_db(KNOWLEDGE_BASE_PATH, PERSIST_DIRECTORY)
        if vectordb and model:
            retriever = vectordb.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=model,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt_template},  # Использование кастомного промпта
                return_source_documents=False
            )
            bot.reply_to(message, "База знаний успешно обновлена.")
            logger.info(f"База знаний была обновлена администратором {message.from_user.id}.")
        else:
            bot.reply_to(message, "Произошла ошибка при обновлении базы знаний.")
    except Exception as e:
        bot.reply_to(message, "Произошла ошибка при обновлении базы знаний.")
        logger.error(f"Ошибка при обновлении базы знаний: {e}")

# Обработчик всех остальных текстовых сообщений (опционально)
@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, "Извините, я не понимаю эту команду. Используйте /iea для вопросов или /start для начала.")

def handle_polling_errors():
    while True:
        try:
            logger.info("Бот запускается...")
            bot.infinity_polling()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Бот остановлен.")
            break  # Прекращаем цикл при завершении программы
        except Exception as e:
            logger.error(f"Произошла ошибка в polling: {e}", exc_info=True)
            logger.info("Перезапуск бота через 15 секунд...")
            time.sleep(15)  # Ожидание перед перезапуском

if __name__ == '__main__':
    try:
        handle_polling_errors()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен.")
    except Exception as e:
        logger.exception(f"Необработанная ошибка: {e}")
