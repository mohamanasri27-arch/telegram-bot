"""Telegram entry point and handlers for the FA<->EN translation bot.

Text messages are translated directly. Voice messages are first transcribed to
Persian text locally, then translated, so the user gets both the clean Persian
transcript and the English version.
"""
import logging
import os
import tempfile

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from transcriber import Transcriber, TranscriptionError
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
    "کاری که این بات انجام می‌ده:\n\n"
    "🎤 *ویس فارسی بفرست* → متن فارسی تمیزشده + ترجمه‌ی انگلیسی رو تحویل می‌گیری.\n"
    "✍️ *متن فارسی بفرست* → ترجمه‌ی انگلیسی می‌گیری.\n"
    "✍️ *متن انگلیسی بفرست* → ترجمه‌ی فارسی می‌گیری.\n\n"
    "برای راهنمای کامل: /help"
)

HELP_MESSAGE = (
    "📖 راهنمای استفاده:\n\n"
    "🎤 ویس فارسی:\n"
    "راحت حرف بزن و ویس رو بفرست. بات اول حرفت رو به متن فارسی تبدیل می‌کنه، "
    "بعد ترجمه‌ی انگلیسی‌اش رو می‌فرسته. متن انگلیسی رو می‌تونی مستقیم برای "
    "هوش مصنوعی کپی کنی.\n\n"
    "✍️ متن:\n"
    "- پیام فارسی بفرست → انگلیسی می‌گیری.\n"
    "- پیام انگلیسی بفرست → فارسی می‌گیری.\n"
    "- بلاک‌های کد (```...```) و کد inline (`...`) دست‌نخورده می‌مونن.\n\n"
    "⏱ نکته: پردازش ویس بسته به طول اون و سرعت سیستم، چند ثانیه تا نیم دقیقه "
    "طول می‌کشه. اولین ویس کمی بیشتر طول می‌کشه چون مدل باید یک‌بار دانلود بشه.\n\n"
    "دستورات:\n"
    "/start - شروع و پیام خوش‌آمد\n"
    "/help - نمایش همین راهنما"
)

ERROR_MESSAGE = "متأسفم، در ترجمه‌ی پیام مشکلی پیش اومد. لطفاً چند لحظه دیگه دوباره امتحان کنید."
VOICE_ERROR_MESSAGE = "متأسفم، نتونستم ویس رو پردازش کنم. لطفاً دوباره امتحان کنید."
TOO_LONG_MESSAGE = "پیام شما طولانی‌تر از حد مجاز (۴۰۰۰ کاراکتر) هست. لطفاً متن کوتاه‌تری بفرستید."
VOICE_PROCESSING_MESSAGE = "🎧 در حال گوش دادن به ویس شما..."
EMPTY_TRANSCRIPT_MESSAGE = "چیزی توی ویس تشخیص ندادم. لطفاً واضح‌تر صحبت کنید و دوباره بفرستید."

translator = Translator()
transcriber = Transcriber()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(START_MESSAGE, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(HELP_MESSAGE)


async def _send_long(message, text: str) -> None:
    """Send text as-is, splitting it when it exceeds Telegram's message limit."""
    for start_index in range(0, len(text), TELEGRAM_MAX_LENGTH):
        await message.reply_text(text[start_index:start_index + TELEGRAM_MAX_LENGTH])


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
        logger.exception("Unhandled error while handling text message")
        await update.message.reply_text(ERROR_MESSAGE)
        return

    if not translated:
        await update.message.reply_text(ERROR_MESSAGE)
        return

    await _send_long(update.message, translated)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    voice = update.message.voice or update.message.audio
    if not voice:
        return

    status_message = await update.message.reply_text(VOICE_PROCESSING_MESSAGE)

    audio_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_file:
            audio_path = temp_file.name

        telegram_file = await voice.get_file()
        await telegram_file.download_to_drive(audio_path)

        transcript = await transcriber.transcribe(audio_path)

        if not transcript:
            await status_message.edit_text(EMPTY_TRANSCRIPT_MESSAGE)
            return

        await status_message.edit_text("📝 متن فارسی:")
        await _send_long(update.message, transcript)

        translated = await translator.translate(transcript)
        if translated:
            await update.message.reply_text("🌐 ترجمه‌ی انگلیسی:")
            await _send_long(update.message, translated)
        else:
            await update.message.reply_text(ERROR_MESSAGE)

    except (TranscriptionError, TranslationError):
        await status_message.edit_text(VOICE_ERROR_MESSAGE)
    except Exception:
        logger.exception("Unhandled error while handling voice message")
        await status_message.edit_text(VOICE_ERROR_MESSAGE)
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


def main() -> None:
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not telegram_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN environment variable is not set")

    app = ApplicationBuilder().token(telegram_token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Bot started")
    app.run_polling()


if __name__ == "__main__":
    main()
