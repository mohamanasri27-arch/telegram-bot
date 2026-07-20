"""Telegram bot that translates Persian text to English and English text to Persian."""
import logging
import os
import re

from deep_translator import GoogleTranslator
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, CommandHandler, filters

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

PERSIAN_RE = re.compile(r"[؀-ۿ]")


def is_persian(text: str) -> bool:
    return bool(PERSIAN_RE.search(text))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "سلام! هر متنی به فارسی بفرستی برات به انگلیسی ترجمه می‌کنم، "
        "و هر متنی به انگلیسی بفرستی برات به فارسی ترجمه می‌کنم."
    )


async def translate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text
    target = "en" if is_persian(text) else "fa"

    try:
        result = GoogleTranslator(source="auto", target=target).translate(text)
    except Exception:
        logger.exception("Translation failed")
        await update.message.reply_text("خطا در ترجمه، لطفاً دوباره امتحان کن.")
        return

    await update.message.reply_text(result)


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_BOT_TOKEN environment variable is not set")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, translate))

    logger.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
