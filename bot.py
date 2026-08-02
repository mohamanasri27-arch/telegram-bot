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
from telegram.error import Conflict, TimedOut
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

# Telegram's defaults are ~5s, which is not enough to pull a voice file over a
# slow or filtered connection. These are deliberately generous.
NETWORK_TIMEOUT = 60.0
MEDIA_TIMEOUT = 300.0

MAX_VOICE_DURATION_SECONDS = 180  # 3 minutes

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
VOICE_TOO_LONG_MESSAGE = (
    "این ویس {duration} ثانیه‌ست و از حد مجاز (۳ دقیقه) بیشتره. "
    "لطفاً به چند ویس کوتاه‌تر تقسیمش کنید."
)
VOICE_TIMEOUT_MESSAGE = (
    "دانلود ویس از سرور تلگرام خیلی طول کشید. لطفاً اینترنت‌تون رو چک کنید و "
    "دوباره بفرستید. اگر ویس طولانی بود، کوتاه‌ترش کنید."
)
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

    duration = voice.duration or 0
    if duration > MAX_VOICE_DURATION_SECONDS:
        await update.message.reply_text(VOICE_TOO_LONG_MESSAGE.format(duration=duration))
        return

    status_message = await update.message.reply_text(VOICE_PROCESSING_MESSAGE)

    audio_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_file:
            audio_path = temp_file.name

        telegram_file = await voice.get_file(
            read_timeout=MEDIA_TIMEOUT,
            connect_timeout=NETWORK_TIMEOUT,
        )
        await telegram_file.download_to_drive(
            audio_path,
            read_timeout=MEDIA_TIMEOUT,
            connect_timeout=NETWORK_TIMEOUT,
        )

        await status_message.edit_text(
            f"✍️ در حال تبدیل {duration} ثانیه صدا به متن... (کمی طول می‌کشه)"
        )

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
    except TimedOut:
        logger.warning("Timed out downloading the voice file from Telegram")
        await status_message.edit_text(VOICE_TIMEOUT_MESSAGE)
    except Exception:
        logger.exception("Unhandled error while handling voice message")
        await status_message.edit_text(VOICE_ERROR_MESSAGE)
    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)


CONFLICT_EXIT_CODE = 3
_conflict_detected = False


async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Stop cleanly on Conflict instead of letting the restart loop spin forever."""
    global _conflict_detected

    if isinstance(context.error, Conflict):
        _conflict_detected = True
        logger.error(
            "\n"
            "============================================================\n"
            "  ANOTHER COPY OF THIS BOT IS ALREADY RUNNING.\n"
            "  Telegram allows only one instance per bot token.\n"
            "  Close every other bot window, then start this one again.\n"
            "============================================================"
        )
        context.application.stop_running()
        return

    logger.error("Update caused an error", exc_info=context.error)


async def _preload_model(application) -> None:
    """Warm the Whisper model at startup so the first voice message isn't slow."""
    logger.info("Preparing the speech model, please wait...")
    try:
        await transcriber.load()
        logger.info("=== BOT IS READY - you can send messages now ===")
    except Exception:
        logger.exception("Could not preload the speech model; will retry on first voice message")


def main() -> None:
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not telegram_token:
        raise SystemExit("TELEGRAM_BOT_TOKEN environment variable is not set")

    app = (
        ApplicationBuilder()
        .token(telegram_token)
        .connect_timeout(NETWORK_TIMEOUT)
        .read_timeout(NETWORK_TIMEOUT)
        .write_timeout(NETWORK_TIMEOUT)
        .media_write_timeout(MEDIA_TIMEOUT)
        .pool_timeout(NETWORK_TIMEOUT)
        .post_init(_preload_model)
        .build()
    )
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_error_handler(_on_error)

    logger.info("Connecting to Telegram and loading the speech model...")
    app.run_polling()

    if _conflict_detected:
        # Signal run.bat not to restart us into the same collision.
        raise SystemExit(CONFLICT_EXIT_CODE)


if __name__ == "__main__":
    main()
