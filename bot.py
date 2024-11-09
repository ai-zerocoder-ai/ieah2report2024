# bot.py

import logging
import os
import threading
import time

from openai import OpenAI  # Импортируем класс OpenAI из библиотеки
import telebot
from telebot import types

from config import TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, KNOWLEDGE_BASE_PATH, ADMIN_ID

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Инициализация Telegram бота
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

# Загрузка базы знаний
def load_knowledge_base_md(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Ошибка загрузки базы знаний: {e}")
        return ""

knowledge_base_md = load_knowledge_base_md(KNOWLEDGE_BASE_PATH)

# Инициализация OpenAI API с использованием новой библиотеки
client = OpenAI(
    api_key=OPENAI_API_KEY,
)

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

# Функция генерации ответа через OpenAI
def sync_generate_response(user_input, knowledge_text):
    system_prompt = (
        "Вы — полезный ассистент, который отвечает на вопросы пользователя, используя предоставленную базу знаний. "
        "Если необходимой информации нет в базе знаний, вежливо сообщите об этом."
    )
    user_prompt = f"Вопрос пользователя: {user_input}\n\nБаза знаний:\n{knowledge_text}\n\nОтвет ассистента:"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Убедитесь, что модель указана правильно
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
        # Предполагается, что структура ответа такая же, как и раньше
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Ошибка при вызове OpenAI API: {e}")
        return "Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже."

# Обработчик команды /iea
@bot.message_handler(commands=['iea'])
def handle_iea_command(message):
    user_input = message.text.partition(' ')[2].strip()
    if not user_input:
        bot.reply_to(message, "Пожалуйста, введите запрос после команды /iea. Пример:\n/iea Каков текущий спрос на водород?")
        return

    logger.info(f"Получен запрос от пользователя {message.from_user.id}: {user_input}")

    def process_and_reply():
        try:
            response = sync_generate_response(user_input, knowledge_base_md)
            bot.send_message(message.chat.id, response, parse_mode='Markdown')
        except Exception as e:
            logger.exception("Ошибка при обработке команды /iea:")
            bot.send_message(message.chat.id, "Произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте позже.")

    # Запуск обработки в отдельном потоке, чтобы не блокировать основной поток бота
    threading.Thread(target=process_and_reply).start()


# Обработчик команды /reload_kb (только для администраторов)
@bot.message_handler(commands=['reload_kb'])
def reload_knowledge_base_command(message):
    if message.from_user.id not in ADMIN_IDS:
        bot.reply_to(message, "У вас нет прав на выполнение этой команды.")
        return

    try:
        global knowledge_base_md
        knowledge_base_md = load_knowledge_base_md(KNOWLEDGE_BASE_PATH)
        bot.reply_to(message, "База знаний успешно обновлена.")
        logger.info(f"База знаний была обновлена администратором {message.from_user.id}.")
    except Exception as e:
        bot.reply_to(message, "Произошла ошибка при обновлении базы знаний.")
        logger.error(f"Ошибка при обновлении базы знаний: {e}")

# Обработчик всех остальных текстовых сообщений (опционально)
@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, "Извините, я не понимаю эту команду. Используйте /iea для вопросов или /start для начала.")

# Функция для обработки ошибок при опросе
def handle_polling_errors():
    while True:
        try:
            logger.info("Бот запускается...")
            bot.infinity_polling()
        except Exception as e:
            logger.error(f"Произошла ошибка в polling: {e}")
            logger.info("Перезапуск бота через 15 секунд...")
            time.sleep(15)  # Ожидание перед перезапуском

if __name__ == '__main__':
    try:
        logger.info("Бот запускается...")
        bot.infinity_polling(timeout=20, long_polling_timeout=20)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Бот остановлен.")
    except Exception as e:
        logger.exception(f"Необработанная ошибка: {e}")

