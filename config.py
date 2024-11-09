# config.py

import os
from dotenv import load_dotenv

# Загрузка переменных окружения из .env файла
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
KNOWLEDGE_BASE_PATH = os.getenv('KNOWLEDGE_BASE_PATH', 'knowledge_base.md')
ADMIN_ID = os.getenv('ADMIN_ID')

# Проверка наличия обязательных переменных
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("Отсутствует TELEGRAM_BOT_TOKEN в переменных окружения.")
if not OPENAI_API_KEY:
    raise ValueError("Отсутствует OPENAI_API_KEY в переменных окружения.")
