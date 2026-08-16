import asyncio
import html
import io
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable
from urllib.parse import urljoin

import aiohttp
from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ChatMemberStatus, ChatType
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.filters import Command, CommandStart
from aiogram.types import BotCommand, Message, MessageEntity, MessageOriginChannel
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

# Главный источник A.
TARGET_CHANNEL_USERNAME = "ibragimmansurov_blog"
TARGET_CHANNEL_PREVIEW = f"https://t.me/s/{TARGET_CHANNEL_USERNAME}"

# Удаляются ТОЛЬКО ссылки / @username из этого точного списка.
# Любые другие t.me-ссылки бот не трогает.
BLOCKED_TELEGRAM_USERNAMES = {
    "ibragimmansurov_blog",
    "manager_ibragimmansurov",
}

# Текстовый кэш.
SOURCE_BOOTSTRAP_PAGES = 5
SOURCE_TEXT_CACHE_LIMIT = 500
SOURCE_REFRESH_SECONDS = 30
FUZZY_MIN_LENGTH = 80
FUZZY_THRESHOLD = 0.94

# Медиа-кэш.
SOURCE_MEDIA_CACHE_LIMIT = 500
SOURCE_MEDIA_MAX_DOWNLOAD_BYTES = 6 * 1024 * 1024
SOURCE_MEDIA_CONCURRENCY = 8

# Порог perceptual hash.
# Чем меньше число, тем строже сравнение.
MEDIA_HASH_MAX_SINGLE_DISTANCE = 10
MEDIA_HASH_MAX_TOTAL_DISTANCE = 14

router = Router()
GROUP_TYPES = {ChatType.GROUP, ChatType.SUPERGROUP}
ADMIN_STATUSES = {ChatMemberStatus.CREATOR, ChatMemberStatus.ADMINISTRATOR}

TARGET_CHANNEL_ID: int | None = None

SOURCE_TEXTS: list[str] = []
SOURCE_TEXT_SET: set[str] = set()

SOURCE_MEDIA: list[tuple["ImageFingerprint", ...]] = []
SOURCE_MEDIA_KEYS: set[tuple[int, int]] = set()

SOURCE_LAST_ERROR: str | None = None

ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
WS_RE = re.compile(r"\s+")
CSS_BG_URL_RE = re.compile(
    r"""background-image\s*:\s*url\(\s*(['"]?)(.*?)\1\s*\)""",
    re.IGNORECASE,
)

TELEGRAM_LINK_RE = re.compile(
    r"""(?ix)
    (?:
        https?://
        |tg://
    )?
    (?:
        (?:www\.)?(?:t|telegram)\.me/
        |
        resolve\?domain=
    )
    (?P<username>[a-z0-9_]{5,})
    """
)


@dataclass(frozen=True)
class ImageFingerprint:
    dhash: int
    ahash: int


def normalize_text(value: str | None) -> str:
    if not value:
        return ""

    text = unicodedata.normalize("NFKC", value)
    text = ZERO_WIDTH_RE.sub("", text)
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("ʼ", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )
    return WS_RE.sub(" ", text).strip().casefold()


def forwarded_channel(message: Message):
    origin = message.forward_origin
    if isinstance(origin, MessageOriginChannel):
        return origin.chat
    return None


def message_text(message: Message) -> str:
    return message.text or message.caption or ""


def message_entities(message: Message) -> Iterable[MessageEntity]:
    yield from (message.entities or [])
    yield from (message.caption_entities or [])


def contains_blocked_reference(message: Message) -> bool:
    normalized = normalize_text(message_text(message))

    # @username в обычном видимом тексте.
    for username in BLOCKED_TELEGRAM_USERNAMES:
        if f"@{username.casefold()}" in normalized:
            return True

    # Видимые t.me / telegram.me / tg:// ссылки.
    for match in TELEGRAM_LINK_RE.finditer(normalized):
        if match.group("username").casefold() in BLOCKED_TELEGRAM_USERNAMES:
            return True

    # Скрытые ссылки Telegram: видимый текст любой, URL находится в entity.url.
    for entity in message_entities(message):
        if not entity.url:
            continue

        normalized_url = normalize_text(entity.url)
        for match in TELEGRAM_LINK_RE.finditer(normalized_url):
            if match.group("username").casefold() in BLOCKED_TELEGRAM_USERNAMES:
                return True

    return False


def is_source_text_copy(text: str) -> tuple[bool, str]:
    candidate = normalize_text(text)
    if not candidate:
        return False, ""

    if candidate in SOURCE_TEXT_SET:
        return True, "exact source text"

    # Для коротких сообщений fuzzy отключён, чтобы избежать случайных удалений.
    if len(candidate) < FUZZY_MIN_LENGTH:
        return False, ""

    # Сильное вложение: скопировали почти весь пост и что-то добавили/убрали.
    for source in SOURCE_TEXTS:
        if len(source) < FUZZY_MIN_LENGTH:
            continue

        shorter = min(len(candidate), len(source))
        longer = max(len(candidate), len(source))
        if shorter / longer >= 0.82 and (candidate in source or source in candidate):
            return True, "source text containment"

    # Почти точная копия длинного текста.
    for source in SOURCE_TEXTS:
        if len(source) < FUZZY_MIN_LENGTH:
            continue

        length_ratio = min(len(candidate), len(source)) / max(len(candidate), len(source))
        if length_ratio < 0.78:
            continue

        ratio = SequenceMatcher(None, candidate, source, autojunk=False).ratio()
        if ratio >= FUZZY_THRESHOLD:
            return True, f"source text similarity={ratio:.3f}"

    return False, ""


def add_source_texts(texts: list[str]) -> int:
    global SOURCE_TEXTS, SOURCE_TEXT_SET

    added = 0
    for raw in texts:
        normalized = normalize_text(raw)
        if not normalized or normalized in SOURCE_TEXT_SET:
            continue

        SOURCE_TEXTS.append(normalized)
        SOURCE_TEXT_SET.add(normalized)
        added += 1

    if len(SOURCE_TEXTS) > SOURCE_TEXT_CACHE_LIMIT:
        SOURCE_TEXTS = SOURCE_TEXTS[-SOURCE_TEXT_CACHE_LIMIT:]
        SOURCE_TEXT_SET = set(SOURCE_TEXTS)

    return added


def _center_crop(image: Image.Image, fraction: float) -> Image.Image:
    width, height = image.size
    crop_w = max(1, int(width * fraction))
    crop_h = max(1, int(height * fraction))
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    return image.crop((left, top, left + crop_w, top + crop_h))


def _dhash(image: Image.Image) -> int:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    value = 0

    for row in range(8):
        offset = row * 9
        for col in range(8):
            value <<= 1
            if pixels[offset + col] > pixels[offset + col + 1]:
                value |= 1

    return value


def _ahash(image: Image.Image) -> int:
    gray = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.getdata())
    average = sum(pixels) / len(pixels)

    value = 0
    for pixel in pixels:
        value <<= 1
        if pixel >= average:
            value |= 1

    return value


def fingerprint_image_bytes(data: bytes) -> tuple[ImageFingerprint, ...]:
    try:
        with Image.open(io.BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        return ()

    # Оригинал + небольшие center-crop варианты.
    # Это помогает пережить небольшое кадрирование/пережатие Telegram.
    variants = [
        image,
        _center_crop(image, 0.96),
        _center_crop(image, 0.90),
    ]

    result: list[ImageFingerprint] = []
    seen: set[tuple[int, int]] = set()

    for variant in variants:
        fp = ImageFingerprint(
            dhash=_dhash(variant),
            ahash=_ahash(variant),
        )
        key = (fp.dhash, fp.ahash)
        if key not in seen:
            seen.add(key)
            result.append(fp)

    return tuple(result)


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def fingerprints_match(
    candidate: tuple[ImageFingerprint, ...],
    source: tuple[ImageFingerprint, ...],
) -> bool:
    for left in candidate:
        for right in source:
            d_distance = _hamming(left.dhash, right.dhash)
            a_distance = _hamming(left.ahash, right.ahash)

            if (
                d_distance <= MEDIA_HASH_MAX_SINGLE_DISTANCE
                and a_distance <= MEDIA_HASH_MAX_SINGLE_DISTANCE
                and d_distance + a_distance <= MEDIA_HASH_MAX_TOTAL_DISTANCE
            ):
                return True

    return False


def source_media_matches(candidate: tuple[ImageFingerprint, ...]) -> bool:
    if not candidate:
        return False

    return any(fingerprints_match(candidate, source) for source in SOURCE_MEDIA)


def _extract_background_url(style: str | None) -> str | None:
    if not style:
        return None

    match = CSS_BG_URL_RE.search(style)
    if not match:
        return None

    return html.unescape(match.group(2).strip())


def extract_message_media_urls(wrap) -> list[str]:
    """
    Берём только media-preview самого поста, не аватар автора канала.
    Для фото это background-image, для видео — poster/thumbnail.
    """
    urls: list[str] = []

    # Фото / превью документов / превью видео, которые Telegram кладёт в CSS.
    for node in wrap.select(
        '[class*="tgme_widget_message_photo"], '
        '[class*="tgme_widget_message_video_thumb"], '
        '[class*="tgme_widget_message_document_thumb"], '
        '[class*="tgme_widget_message_media"]'
    ):
        class_text = " ".join(node.get("class", []))

        # Никогда не хэшируем аватар автора.
        if "user_photo" in class_text or "author_photo" in class_text:
            continue

        bg_url = _extract_background_url(node.get("style"))
        if bg_url:
            urls.append(urljoin(TARGET_CHANNEL_PREVIEW, bg_url))

        src = node.get("src")
        if src and not str(src).lower().endswith((".mp4", ".webm", ".mov")):
            urls.append(urljoin(TARGET_CHANNEL_PREVIEW, html.unescape(src)))

    # Для обычных видео и video message Telegram web-preview обычно имеет poster.
    for video in wrap.select("video"):
        poster = video.get("poster")
        if poster:
            urls.append(urljoin(TARGET_CHANNEL_PREVIEW, html.unescape(poster)))

    # На случай, если poster лежит не на <video>, а на media-элементе.
    for node in wrap.select("[poster]"):
        poster = node.get("poster")
        if poster:
            urls.append(urljoin(TARGET_CHANNEL_PREVIEW, html.unescape(poster)))

    # Убираем дубли, сохраняя порядок.
    return list(dict.fromkeys(urls))


async def fetch_preview_page(
    session: aiohttp.ClientSession,
    before: int | None = None,
) -> tuple[list[str], int | None, list[str]]:
    url = TARGET_CHANNEL_PREVIEW
    if before is not None:
        url = f"{url}?before={before}"

    async with session.get(url, allow_redirects=True) as response:
        response.raise_for_status()
        html_text = await response.text()

    soup = BeautifulSoup(html_text, "html.parser")

    texts: list[str] = []
    ids: list[int] = []
    media_urls: list[str] = []

    for wrap in soup.select(".tgme_widget_message_wrap"):
        message_node = wrap.select_one(".tgme_widget_message")
        if message_node:
            data_post = message_node.get("data-post")
            if data_post and "/" in data_post:
                try:
                    ids.append(int(data_post.rsplit("/", 1)[1]))
                except ValueError:
                    pass

        text_node = wrap.select_one(".tgme_widget_message_text")
        if text_node:
            value = text_node.get_text("\n", strip=True)
            if value:
                texts.append(value)

        media_urls.extend(extract_message_media_urls(wrap))

    return texts, min(ids) if ids else None, list(dict.fromkeys(media_urls))


async def download_preview_image(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
) -> tuple[ImageFingerprint, ...]:
    async with semaphore:
        try:
            async with session.get(url, allow_redirects=True) as response:
                response.raise_for_status()

                content_type = (response.headers.get("Content-Type") or "").lower()
                if content_type and not content_type.startswith("image/"):
                    return ()

                data = await response.content.read(SOURCE_MEDIA_MAX_DOWNLOAD_BYTES + 1)
                if len(data) > SOURCE_MEDIA_MAX_DOWNLOAD_BYTES:
                    return ()

            return await asyncio.to_thread(fingerprint_image_bytes, data)
        except Exception as exc:
            logging.debug("Could not hash source media %s: %s", url, exc)
            return ()


def add_source_media_fingerprints(
    fingerprints: list[tuple[ImageFingerprint, ...]],
) -> int:
    global SOURCE_MEDIA, SOURCE_MEDIA_KEYS

    added = 0

    for variants in fingerprints:
        if not variants:
            continue

        primary = variants[0]
        key = (primary.dhash, primary.ahash)
        if key in SOURCE_MEDIA_KEYS:
            continue

        SOURCE_MEDIA.append(variants)
        SOURCE_MEDIA_KEYS.add(key)
        added += 1

    if len(SOURCE_MEDIA) > SOURCE_MEDIA_CACHE_LIMIT:
        SOURCE_MEDIA = SOURCE_MEDIA[-SOURCE_MEDIA_CACHE_LIMIT:]
        SOURCE_MEDIA_KEYS = {
            (variants[0].dhash, variants[0].ahash)
            for variants in SOURCE_MEDIA
            if variants
        }

    return added


async def load_source_media(
    session: aiohttp.ClientSession,
    urls: list[str],
) -> int:
    if not urls:
        return 0

    semaphore = asyncio.Semaphore(SOURCE_MEDIA_CONCURRENCY)

    results = await asyncio.gather(
        *(download_preview_image(session, url, semaphore) for url in urls),
        return_exceptions=False,
    )

    return add_source_media_fingerprints(list(results))


def source_http_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/151 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }


async def bootstrap_source_cache() -> None:
    global SOURCE_LAST_ERROR

    timeout = aiohttp.ClientTimeout(total=20)

    try:
        async with aiohttp.ClientSession(
            timeout=timeout,
            headers=source_http_headers(),
        ) as session:
            before = None
            all_texts: list[str] = []
            all_media_urls: list[str] = []

            for _ in range(SOURCE_BOOTSTRAP_PAGES):
                texts, next_before, media_urls = await fetch_preview_page(session, before)
                all_texts.extend(texts)
                all_media_urls.extend(media_urls)

                if next_before is None or next_before == before:
                    break
                before = next_before

            text_added = add_source_texts(all_texts)
            media_added = await load_source_media(
                session,
                list(dict.fromkeys(all_media_urls)),
            )

        SOURCE_LAST_ERROR = None
        logging.info(
            "Source cache bootstrapped: texts=%s (+%s), media=%s (+%s)",
            len(SOURCE_TEXTS),
            text_added,
            len(SOURCE_MEDIA),
            media_added,
        )
    except Exception as exc:
        SOURCE_LAST_ERROR = f"{type(exc).__name__}: {exc}"
        logging.warning("Source bootstrap failed: %s", SOURCE_LAST_ERROR)


async def refresh_source_cache_once() -> None:
    global SOURCE_LAST_ERROR

    timeout = aiohttp.ClientTimeout(total=20)

    try:
        async with aiohttp.ClientSession(
            timeout=timeout,
            headers=source_http_headers(),
        ) as session:
            texts, _, media_urls = await fetch_preview_page(session)

            text_added = add_source_texts(texts)
            media_added = await load_source_media(session, media_urls)

        SOURCE_LAST_ERROR = None

        if text_added or media_added:
            logging.info(
                "Source cache refreshed: texts +%s, media +%s; totals=%s/%s",
                text_added,
                media_added,
                len(SOURCE_TEXTS),
                len(SOURCE_MEDIA),
            )
    except Exception as exc:
        SOURCE_LAST_ERROR = f"{type(exc).__name__}: {exc}"
        logging.warning("Source refresh failed: %s", SOURCE_LAST_ERROR)


async def source_refresh_loop() -> None:
    while True:
        await asyncio.sleep(SOURCE_REFRESH_SECONDS)
        await refresh_source_cache_once()


def downloadable_media_for_message(message: Message):
    """
    Возвращает картинку/thumbnail, по которой можно сравнить сообщение.
    Видео целиком не скачиваем — берём thumbnail/cover.
    """

    # Фото: берём самую большую версию.
    if message.photo:
        return message.photo[-1]

    # Круглое видео message.
    if message.video_note and message.video_note.thumbnail:
        return message.video_note.thumbnail

    # Обычное видео.
    if message.video:
        if message.video.thumbnail:
            return message.video.thumbnail
        if message.video.cover:
            return message.video.cover[-1]

    # GIF / animation.
    if message.animation and message.animation.thumbnail:
        return message.animation.thumbnail

    # Если фото/видео отправили "как файл", у Telegram часто есть thumbnail.
    if message.document:
        mime_type = (message.document.mime_type or "").lower()

        if mime_type.startswith("image/"):
            # Само изображение даёт более точный отпечаток, если Bot API позволяет скачать.
            if not message.document.file_size or message.document.file_size <= 20 * 1024 * 1024:
                return message.document

        if message.document.thumbnail:
            return message.document.thumbnail

    return None


async def message_media_fingerprint(
    message: Message,
    bot: Bot,
) -> tuple[ImageFingerprint, ...]:
    media = downloadable_media_for_message(message)
    if media is None:
        return ()

    try:
        buffer = await bot.download(media)
        if buffer is None:
            return ()

        data = buffer.getvalue()
        return await asyncio.to_thread(fingerprint_image_bytes, data)
    except Exception as exc:
        logging.warning(
            "Could not fingerprint group media message=%s: %s",
            message.message_id,
            exc,
        )
        return ()


async def delete_message_safely(
    message: Message,
    bot: Bot,
    reason: str,
) -> None:
    try:
        await bot.delete_message(message.chat.id, message.message_id)
        logging.info(
            "Deleted: group=%s message=%s user=%s reason=%s",
            message.chat.id,
            message.message_id,
            message.from_user.id if message.from_user else None,
            reason,
        )
    except TelegramForbiddenError:
        logging.error("No permission to delete in group %s", message.chat.id)
    except TelegramBadRequest as exc:
        logging.warning("Telegram rejected deletion: %s", exc)


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Я защищаю группу от контента из запрещённого источника.\n\n"
        "Удаляю:\n"
        f"1. Forward из @{TARGET_CHANNEL_USERNAME};\n"
        "2. точные запрещённые @username / t.me ссылки;\n"
        "3. текстовые копии и почти точные копии постов;\n"
        "4. повторно загруженные фото;\n"
        "5. повторно загруженные обычные и круглые видео по thumbnail/кадру.\n\n"
        "Проверка: /status"
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await cmd_start(message)


@router.message(Command("status"))
async def cmd_status(message: Message, bot: Bot) -> None:
    if message.chat.type not in GROUP_TYPES:
        await message.answer("Команду /status нужно использовать внутри группы.")
        return

    me = await bot.get_me()

    try:
        member = await bot.get_chat_member(message.chat.id, me.id)
    except (TelegramBadRequest, TelegramForbiddenError) as exc:
        await message.answer(f"Не удалось проверить мои права: {exc}")
        return

    is_admin = member.status in ADMIN_STATUSES
    can_delete = bool(getattr(member, "can_delete_messages", False)) if is_admin else False

    target_status = (
        f"@{TARGET_CHANNEL_USERNAME}: ID {TARGET_CHANNEL_ID}"
        if TARGET_CHANNEL_ID is not None
        else f"@{TARGET_CHANNEL_USERNAME}: ID не определён"
    )

    cache_status = (
        f"Кэш текста: {len(SOURCE_TEXTS)}\n"
        f"Кэш медиа: {len(SOURCE_MEDIA)}"
    )

    if SOURCE_LAST_ERROR:
        cache_status += f"\nПоследняя ошибка источника: {SOURCE_LAST_ERROR}"

    await message.answer(
        "Статус:\n"
        f"Администратор: {'да' if is_admin else 'нет'}\n"
        f"Может удалять сообщения: {'да' if can_delete else 'нет'}\n"
        f"{target_status}\n"
        f"{cache_status}\n\n"
        + (
            "Модерация готова."
            if is_admin and can_delete
            else "Сначала выдай боту право удаления сообщений."
        )
    )


@router.message()
async def moderate_messages(message: Message, bot: Bot) -> None:
    if message.chat.type not in GROUP_TYPES:
        return

    # 1. Настоящий Telegram Forward из A.
    channel = forwarded_channel(message)
    if channel is not None:
        blocked_forward = (
            TARGET_CHANNEL_ID is not None and channel.id == TARGET_CHANNEL_ID
        )

        if not blocked_forward and channel.username:
            blocked_forward = (
                channel.username.casefold() == TARGET_CHANNEL_USERNAME.casefold()
            )

        if blocked_forward:
            await delete_message_safely(message, bot, "forward from blocked channel")
            return

    # 2. ТОЛЬКО конкретные запрещённые Telegram usernames/links.
    if contains_blocked_reference(message):
        await delete_message_safely(message, bot, "blocked Telegram username/link")
        return

    # 3. Копия текста/подписи.
    copied, reason = is_source_text_copy(message_text(message))
    if copied:
        await delete_message_safely(message, bot, reason)
        return

    # 4. Фото/видео/video-note, скачанные из A и загруженные заново.
    if SOURCE_MEDIA:
        candidate_fingerprint = await message_media_fingerprint(message, bot)

        if candidate_fingerprint and source_media_matches(candidate_fingerprint):
            await delete_message_safely(message, bot, "source media visual match")
            return


async def resolve_target_channel(bot: Bot) -> None:
    global TARGET_CHANNEL_ID

    try:
        chat = await bot.get_chat(f"@{TARGET_CHANNEL_USERNAME}")
        TARGET_CHANNEL_ID = chat.id
        logging.info(
            "Target channel resolved: @%s -> %s",
            TARGET_CHANNEL_USERNAME,
            TARGET_CHANNEL_ID,
        )
    except Exception as exc:
        TARGET_CHANNEL_ID = None
        logging.warning(
            "Could not resolve @%s: %s. Username fallback remains enabled.",
            TARGET_CHANNEL_USERNAME,
            exc,
        )


async def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN не задан. Добавь BOT_TOKEN в Railway Variables."
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    bot = Bot(BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)

    await bot.set_my_commands(
        [
            BotCommand(command="status", description="Проверить работу"),
            BotCommand(command="help", description="Инструкция"),
        ]
    )

    await resolve_target_channel(bot)
    await bootstrap_source_cache()

    refresh_task = asyncio.create_task(source_refresh_loop())

    try:
        logging.info(
            "Channel Guard v3 started. Target=@%s texts=%s media=%s",
            TARGET_CHANNEL_USERNAME,
            len(SOURCE_TEXTS),
            len(SOURCE_MEDIA),
        )
        await dp.start_polling(bot)
    finally:
        refresh_task.cancel()
        try:
            await refresh_task
        except asyncio.CancelledError:
            pass
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
