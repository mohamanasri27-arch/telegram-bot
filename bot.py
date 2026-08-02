"""Telegram entry point and handlers for the FA<->EN translation bot.

Text messages are translated directly. Voice messages are first transcribed to
Persian text locally, then translated, so the user gets both the clean Persian
transcript and the English version.
"""
import logging
import os
import tempfile

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import Conflict, TimedOut
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import prompt_format
import settings
import vocabulary
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
    "💡 *نکته:* هر جا کلمه‌ای رو اشتباه شنید، با `/add` بهش یاد بده — "
    "دفعه‌ی بعد درست می‌گه.\n\n"
    "🧠 با `/mode` می‌تونی خروجی رو به شکل پرامپت آماده برای AI بگیری.\n\n"
    "راهنمای کامل: /help"
)

HELP_MESSAGE = (
    "📖 راهنمای استفاده:\n\n"
    "🎤 ویس فارسی بفرست (تا ۳ دقیقه) → متن فارسی + ترجمه‌ی انگلیسی می‌گیری.\n"
    "✍️ متن فارسی → انگلیسی | متن انگلیسی → فارسی\n"
    "بلاک‌های کد (```...```) دست‌نخورده می‌مونن.\n\n"
    "━━━━━━━━━━━━━━━━━━\n"
    "📚 آموزش دادن واژه به بات\n\n"
    "هر جا بات کلمه‌ای رو اشتباه شنید یا بد ترجمه کرد، بهش یاد بده:\n\n"
    "`/add ابر آروان = ArvanCloud`\n\n"
    "از اون به بعد هم درست می‌شنوه، هم دقیقاً همون رو ترجمه می‌کنه.\n"
    "برای دیدن واژه‌های اضافه‌شده: /terms\n\n"
    "━━━━━━━━━━━━━━━━━━\n"
    "⚙️ تنظیمات\n\n"
    "/mode — جابه‌جایی بین دو حالت خروجی:\n"
    "  • *عادی* — ترجمه‌ی ساده\n"
    "  • *پرامپت* — خروجی مرتب‌شده با Context و Task، آماده برای دادن به AI\n\n"
    "/accuracy — جابه‌جایی بین دو مدل تشخیص گفتار:\n"
    "  • *سریع* — پیش‌فرض، چند ثانیه\n"
    "  • *دقیق* — کیفیت بالاتر، حدود ۳ برابر کندتر\n\n"
    "/fillers — حذف یا نگه داشتن کلمات پرکننده («خب»، «یعنی»، ...)\n"
    "/settings — نمایش وضعیت فعلی\n\n"
    "━━━━━━━━━━━━━━━━━━\n"
    "/start — پیام خوش‌آمد\n"
    "/help — همین راهنما"
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


ADD_USAGE_MESSAGE = (
    "برای اضافه کردن واژه، این شکلی بنویسید:\n\n"
    "`/add کلمه فارسی = English Term`\n\n"
    "مثال:\n"
    "`/add ابر آروان = ArvanCloud`"
)


async def add_term(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Teach the bot a term without touching any file on disk."""
    raw = " ".join(context.args or "")
    if "=" not in raw:
        await update.message.reply_text(ADD_USAGE_MESSAGE, parse_mode="Markdown")
        return

    persian, english = (part.strip() for part in raw.split("=", 1))
    if not persian or not english:
        await update.message.reply_text(ADD_USAGE_MESSAGE, parse_mode="Markdown")
        return

    try:
        vocabulary.add_term(persian, english)
    except OSError:
        logger.exception("Could not write the new term")
        await update.message.reply_text("نتونستم واژه رو ذخیره کنم. دسترسی نوشتن روی فایل نیست.")
        return

    translator.reload_glossary()
    transcriber.reload_hotwords()

    await update.message.reply_text(
        f"✅ یاد گرفتم:\n\n«{persian}» ← «{english}»\n\n"
        "از این به بعد همین رو استفاده می‌کنم. برای دیدن همه: /terms"
    )


async def list_terms(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    mine = vocabulary.load_my_terms()
    if not mine:
        await update.message.reply_text(
            "هنوز واژه‌ای اضافه نکردید.\n\n" + ADD_USAGE_MESSAGE, parse_mode="Markdown"
        )
        return

    lines = "\n".join(f"• {p} ← {e}" for p, e in mine.items())
    await _send_long(update.message, f"📚 واژه‌های اضافه‌شده ({len(mine)}):\n\n{lines}")


async def toggle_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    current = settings.get("mode")
    new = settings.MODE_PROMPT if current == settings.MODE_PLAIN else settings.MODE_PLAIN
    settings.set_value("mode", new)

    if new == settings.MODE_PROMPT:
        await update.message.reply_text(
            "🧠 حالت *پرامپت* فعال شد.\n\n"
            "از این به بعد خروجی انگلیسی با بخش‌های Context و Task مرتب می‌شه "
            "تا هوش مصنوعی بهتر منظورت رو بفهمه.",
            parse_mode="Markdown",
        )
    else:
        await update.message.reply_text(
            "📝 حالت *عادی* فعال شد.\n\nخروجی، ترجمه‌ی ساده و بدون بخش‌بندی خواهد بود.",
            parse_mode="Markdown",
        )


async def toggle_fillers(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    new = not settings.get("clean_fillers")
    settings.set_value("clean_fillers", new)
    await update.message.reply_text(
        "🧹 حذف کلمات پرکننده *روشن* شد.\n\n«خب»، «یعنی»، «ببین» و مشابه از متن حذف می‌شن."
        if new
        else "🧹 حذف کلمات پرکننده *خاموش* شد.\n\nمتن دقیقاً همون چیزی می‌مونه که گفتی.",
        parse_mode="Markdown",
    )


async def toggle_accuracy(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    current = settings.get("model")
    new = (
        settings.ACCURACY_HIGH
        if current == settings.ACCURACY_FAST
        else settings.ACCURACY_FAST
    )
    settings.set_value("model", new)

    if new == settings.ACCURACY_HIGH:
        status = await update.message.reply_text(
            "🎯 در حال تغییر به مدل *دقیق*...\n\n"
            "⚠️ *بات تا پایان این کار به هیچ پیامی جواب نمی‌ده* — این طبیعیه، "
            "خراب نشده. بارگذاری مدل کل بات رو موقتاً قفل می‌کنه.\n\n"
            "اگر اولین بار باشه حدود ۳ گیگابایت دانلود می‌شه و بسته به سرعت "
            "اینترنت‌تون چند دقیقه تا نیم ساعت طول می‌کشه.\n\n"
            "پیشرفت رو توی پنجره‌ی اجرا ببینید. وقتی خط "
            "`Whisper model ready` اومد، دوباره جواب می‌ده.",
            parse_mode="Markdown",
        )
    else:
        status = await update.message.reply_text(
            "⚡ در حال برگشت به مدل *سریع*...\n\n"
            "بات چند لحظه جواب نمی‌ده تا مدل عوض بشه.",
            parse_mode="Markdown",
        )

    try:
        await transcriber.switch_model(new)
    except Exception:
        logger.exception("Could not switch model")
        settings.set_value("model", current)
        await status.edit_text(
            "نتونستم مدل رو عوض کنم. احتمالاً دانلود ناموفق بوده. "
            "اتصال اینترنت رو چک کن و دوباره امتحان کن."
        )
        return

    label = "دقیق (کندتر)" if new == settings.ACCURACY_HIGH else "سریع"
    await status.edit_text(f"✅ مدل تشخیص گفتار الان روی حالت *{label}* است.", parse_mode="Markdown")


async def show_settings(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    current = settings.load()
    mode = "پرامپت 🧠" if current["mode"] == settings.MODE_PROMPT else "عادی 📝"
    accuracy = "دقیق 🎯" if current["model"] == settings.ACCURACY_HIGH else "سریع ⚡"
    fillers = "روشن ✅" if current["clean_fillers"] else "خاموش ❌"
    mine = vocabulary.load_my_terms()

    await update.message.reply_text(
        "⚙️ *وضعیت فعلی*\n\n"
        f"حالت خروجی: {mode}   (/mode)\n"
        f"دقت تشخیص: {accuracy}   (/accuracy)\n"
        f"حذف کلمات پرکننده: {fillers}   (/fillers)\n"
        f"واژه‌های اضافه‌شده: {len(mine)}   (/terms)",
        parse_mode="Markdown",
    )


def _result_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("🔁 ترجمه‌ی مجدد", callback_data="retranslate"),
            InlineKeyboardButton("📚 افزودن واژه", callback_data="howto_add"),
        ]]
    )


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    if query.data == "howto_add":
        await query.message.reply_text(ADD_USAGE_MESSAGE, parse_mode="Markdown")
        return

    if query.data == "retranslate":
        source = context.user_data.get("last_persian")
        if not source:
            await query.message.reply_text("متن قبلی رو پیدا نکردم. لطفاً دوباره بفرستش.")
            return
        try:
            translated = await translator.translate(source)
        except Exception:
            logger.exception("Retranslation failed")
            await query.message.reply_text(ERROR_MESSAGE)
            return
        await _deliver_translation(query.message, translated)


async def _deliver_translation(message, translated: str) -> None:
    """Send the English result, formatted according to the current mode."""
    if settings.get("mode") == settings.MODE_PROMPT:
        translated = prompt_format.build(translated)
    await _send_long(message, translated)


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

    context.user_data["last_persian"] = text
    await _deliver_translation(update.message, translated)


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

        context.user_data["last_persian"] = transcript

        await status_message.edit_text("📝 متن فارسی:")
        await _send_long(update.message, transcript)

        translated = await translator.translate(transcript)
        if translated:
            await update.message.reply_text("🌐 ترجمه‌ی انگلیسی:")
            await _deliver_translation(update.message, translated)
            await update.message.reply_text(
                "اگر کلمه‌ای اشتباه بود، بهم یاد بده 👇", reply_markup=_result_keyboard()
            )
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
    app.add_handler(CommandHandler("add", add_term))
    app.add_handler(CommandHandler("terms", list_terms))
    app.add_handler(CommandHandler("mode", toggle_mode))
    app.add_handler(CommandHandler("accuracy", toggle_accuracy))
    app.add_handler(CommandHandler("fillers", toggle_fillers))
    app.add_handler(CommandHandler("settings", show_settings))
    app.add_handler(CallbackQueryHandler(on_button))
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
