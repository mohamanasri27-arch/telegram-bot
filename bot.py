"""Telegram entry point and handlers for the FA<->EN technical translation bot."""
import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from translator import Translator, TranslationError

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_MAX_LENGTH = 4000

START_MESSAGE = (
    "سلام! 👋\n\n"
    "این بات پیام‌های شما رو بین فارسی و انگلیسی ترجمه می‌کنه، مخصوص متن‌های فنی "
    "برنامه‌نویسی و پرامپت‌های هوش مصنوعی.\n\n"
    "فقط کافیه متنت رو بفرستی؛ زبان به‌صورت خودکار تشخیص داده می‌شه و بلاک‌های کد "
    "و اصطلاحات تکنیکال دست‌نخورده باقی می‌مونن.\n\n"
    "برای راهنمای کامل: /help"
)

HELP_MESSAGE = (
    "📖 راهنمای استفاده:\n\n"
    "- یک پیام فارسی بفرستید تا به انگلیسی ترجمه بشه.\n"
    "- یک پیام انگلیسی بفرستید تا به فارسی ترجمه بشه.\n"
    "- بلاک‌های کد (```...```) و کد inline (`...`) عیناً حفظ می‌شن.\n"
    "- اسامی لایبراری‌ها، توابع، متغیرها و اصطلاحات تکنیکال رایج ترجمه نمی‌شن.\n"
    "- حداکثر طول پیام: ۴۰۰۰ کاراکتر.\n\n"
    "دستورات:\n"
    "/start - شروع و پیام خوش‌آمد\n"
    "/help - نمایش همین راهنما"
)

ERROR_MESSAGE = "متأسفم، در ترجمه‌ی پیام مشکلی پیش اومد. لطفاً چند لحظه دیگه دوباره امتحان کنید."
TOO_LONG_MESSAGE = "پیام شما طولانی‌تر از حد مجاز (۴۰۰۰ کاراکتر) هست. لطفاً متن کوتاه‌تری بفرستید."

translator = Translator()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(START_MESSAGE)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_MESSAGE)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text
    if not text:
        return

    if len(text) > TELEGRAM_MAX_LENGTH:
        await update.message.reply_text(TOO_LONG_MESSAGE)
        return

    try:
        translated = await translator.translate(text)
    except TranslationError:
        await update.message.reply_text(ERROR_MESSAGE)
        return
    except Exception:
        logger.exception("Unhandled error while handling message")
        await update.message.reply_text(ERROR_MESSAGE)
        return

    if not translated:
        await update.message.reply_text(ERROR_MESSAGE)
        return

    try:
        await update.message.reply_text(translated, parse_mode=ParseMode.MARKDOWN)
    except Exception:
        logger.warning("Failed to send with Markdown parsing, falling back to plain text")
        await update.message.reply_text(translated)


def main() -> None:
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not telegram_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN environment variable is not set")

    app = ApplicationBuilder().token(telegram_token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
