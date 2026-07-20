"""Translation logic using the Claude API, kept independent of the Telegram layer."""
import asyncio
import logging
import os
import re

from anthropic import Anthropic, APIStatusError, RateLimitError

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-5"
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds

PERSIAN_RE = re.compile(r"[؀-ۿ]")

SYSTEM_PROMPT = """\
تو یک مترجم متخصص بین فارسی و انگلیسی هستی، مخصوص متن‌های فنی برنامه‌نویسی و \
پرامپت‌هایی که برای مدل‌های هوش مصنوعی نوشته می‌شن.

قوانین:
- اگر متن ورودی فارسیه، به انگلیسی ترجمه کن. اگر انگلیسیه، به فارسی ترجمه کن.
- بلاک‌های کد (بین ```) و کد inline (بین `) رو دقیقاً و بدون هیچ تغییری دست‌نخورده نگه دار.
- اسامی لایبراری‌ها، فریمورک‌ها، توابع، کلاس‌ها، متغیرها و مسیرهای فایل رو هرگز ترجمه نکن.
- اصطلاحات تکنیکال رایج (مثل API, endpoint, prompt, token, framework, function, \
variable, repository, deploy, backend, frontend و مشابه اون‌ها) رو در ترجمه به فارسی \
به همون شکل انگلیسی نگه دار، معادل مصنوعی فارسی براشون نساز.
- لحن و سطح تخصصی متن اصلی رو حفظ کن.
- فقط و فقط ترجمه‌ی نهایی رو خروجی بده. هیچ توضیح، مقدمه، یا جمله‌ی اضافه ننویس.
"""


def is_persian(text: str) -> bool:
    return bool(PERSIAN_RE.search(text))


class TranslationError(Exception):
    pass


class Translator:
    def __init__(self, api_key: str | None = None) -> None:
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    async def translate(self, text: str) -> str:
        target_lang = "انگلیسی" if is_persian(text) else "فارسی"
        user_message = f"متن زیر رو به {target_lang} ترجمه کن:\n\n{text}"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = await asyncio.to_thread(
                    self._client.messages.create,
                    model=MODEL,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_message}],
                )
                return "".join(
                    block.text for block in response.content if block.type == "text"
                ).strip()
            except RateLimitError:
                logger.warning("Rate limited by Claude API (attempt %d/%d)", attempt, MAX_RETRIES)
                if attempt == MAX_RETRIES:
                    raise TranslationError("rate_limited") from None
                await asyncio.sleep(RETRY_BASE_DELAY * attempt)
            except APIStatusError:
                logger.exception("Claude API error (attempt %d/%d)", attempt, MAX_RETRIES)
                if attempt == MAX_RETRIES:
                    raise TranslationError("api_error") from None
                await asyncio.sleep(RETRY_BASE_DELAY * attempt)
            except Exception as exc:
                logger.exception("Unexpected translation failure")
                raise TranslationError("unknown") from exc

        raise TranslationError("unknown")
