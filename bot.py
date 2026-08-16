import asyncio
import io
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable

from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ChatMemberStatus, ChatType
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.filters import Command, CommandStart
from aiogram.types import BotCommand, Message, MessageEntity, MessageOriginChannel
from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.sessions import StringSession

load_dotenv()

# ---------------------------------------------------------------------------
# ENV
# ---------------------------------------------------------------------------

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
API_ID_RAW = os.getenv("API_ID", "").strip()
API_HASH = os.getenv("API_HASH", "").strip()
TELETHON_SESSION = os.getenv("TELETHON_SESSION", "").strip()

# ---------------------------------------------------------------------------
# SOURCE A
# ---------------------------------------------------------------------------

TARGET_CHANNEL_USERNAME = "ibragimmansurov_blog"

# Удаляются ТОЛЬКО Telegram usernames/links из этого списка.
# Обычные t.me ссылки не затрагиваются.
BLOCKED_TELEGRAM_USERNAMES = {
    "ibragimmansurov_blog",
    "manager_ibragimmansurov",
}

# Сколько реальных сообщений канала A читать через Telethon.
# 2000 — хороший запас для старых публикаций.
SOURCE_HISTORY_LIMIT = 2000

# Быстрый скан перед запуском polling бота.
SOURCE_QUICK_SCAN_LIMIT = 120

# Сколько последних сообщений канала перепроверять периодически.
SOURCE_REFRESH_LIMIT = 50
SOURCE_REFRESH_SECONDS = 60

SOURCE_TEXT_CACHE_LIMIT = 3000
SOURCE_MEDIA_CACHE_LIMIT = 3000

# ---------------------------------------------------------------------------
# TEXT MATCHING
# ---------------------------------------------------------------------------

FUZZY_MIN_LENGTH = 80
FUZZY_THRESHOLD = 0.94

# ---------------------------------------------------------------------------
# MEDIA MATCHING
# ---------------------------------------------------------------------------

# Чем меньше значения — тем строже совпадение.
MEDIA_HASH_MAX_SINGLE_DISTANCE = 12
MEDIA_HASH_MAX_TOTAL_DISTANCE = 18

# Для видео дополнительно проверяем длительность, когда она известна.
VIDEO_DURATION_TOLERANCE_SECONDS = 3

# Сколько thumbnail-вариантов пробовать у Telethon media.
TELETHON_THUMB_CANDIDATES = (-1, -2, -3, -4, -5)

# ---------------------------------------------------------------------------

router = Router()
GROUP_TYPES = {ChatType.GROUP, ChatType.SUPERGROUP}
ADMIN_STATUSES = {ChatMemberStatus.CREATOR, ChatMemberStatus.ADMINISTRATOR}

TARGET_CHANNEL_ID: int | None = None
TARGET_ENTITY = None

SOURCE_TEXTS: list[str] = []
SOURCE_TEXT_SET: set[str] = set()

SOURCE_MEDIA: list["SourceMediaFingerprint"] = []
SOURCE_MEDIA_KEYS: set[tuple] = set()

SOURCE_LAST_ERROR: str | None = None
TELETHON_CONNECTED = False
HISTORY_SCAN_RUNNING = False
HISTORY_SCAN_DONE = False
HISTORY_SCAN_PROGRESS = 0

ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
WS_RE = re.compile(r"\s+")

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


@dataclass(frozen=True)
class SourceMediaFingerprint:
    kind: str  # "photo" | "video"
    duration: int | None
    variants: tuple[ImageFingerprint, ...]


# ---------------------------------------------------------------------------
# TEXT / LINKS
# ---------------------------------------------------------------------------

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

    # 1. Видимые @username.
    for username in BLOCKED_TELEGRAM_USERNAMES:
        if f"@{username.casefold()}" in normalized:
            return True

    # 2. Видимые Telegram-ссылки.
    for match in TELEGRAM_LINK_RE.finditer(normalized):
        if match.group("username").casefold() in BLOCKED_TELEGRAM_USERNAMES:
            return True

    # 3. Скрытые text_link URL.
    for entity in message_entities(message):
        if not entity.url:
            continue

        normalized_url = normalize_text(entity.url)
        for match in TELEGRAM_LINK_RE.finditer(normalized_url):
            if match.group("username").casefold() in BLOCKED_TELEGRAM_USERNAMES:
                return True

    return False


def add_source_text(text: str | None) -> bool:
    global SOURCE_TEXTS, SOURCE_TEXT_SET

    normalized = normalize_text(text)
    if not normalized or normalized in SOURCE_TEXT_SET:
        return False

    SOURCE_TEXTS.append(normalized)
    SOURCE_TEXT_SET.add(normalized)

    if len(SOURCE_TEXTS) > SOURCE_TEXT_CACHE_LIMIT:
        SOURCE_TEXTS = SOURCE_TEXTS[-SOURCE_TEXT_CACHE_LIMIT:]
        SOURCE_TEXT_SET = set(SOURCE_TEXTS)

    return True


def is_source_text_copy(text: str) -> tuple[bool, str]:
    candidate = normalize_text(text)
    if not candidate:
        return False, ""

    if candidate in SOURCE_TEXT_SET:
        return True, "exact source text"

    if len(candidate) < FUZZY_MIN_LENGTH:
        return False, ""

    # Почти весь исходный пост + небольшая добавка/удаление.
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


# ---------------------------------------------------------------------------
# IMAGE FINGERPRINTING
# ---------------------------------------------------------------------------

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

    # Оригинал + небольшие crop + mirror.
    # Это переживает изменение размера, Telegram-сжатие,
    # небольшое кадрирование и зеркальное отражение.
    base_variants = [
        image,
        _center_crop(image, 0.96),
        _center_crop(image, 0.90),
    ]

    variants = []
    for variant in base_variants:
        variants.append(variant)
        variants.append(ImageOps.mirror(variant))

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


def fingerprint_variants_match(
    left_variants: tuple[ImageFingerprint, ...],
    right_variants: tuple[ImageFingerprint, ...],
) -> bool:
    for left in left_variants:
        for right in right_variants:
            d_distance = _hamming(left.dhash, right.dhash)
            a_distance = _hamming(left.ahash, right.ahash)

            if (
                d_distance <= MEDIA_HASH_MAX_SINGLE_DISTANCE
                and a_distance <= MEDIA_HASH_MAX_SINGLE_DISTANCE
                and d_distance + a_distance <= MEDIA_HASH_MAX_TOTAL_DISTANCE
            ):
                return True

    return False


def media_duration_compatible(
    candidate_duration: int | None,
    source_duration: int | None,
) -> bool:
    if candidate_duration is None or source_duration is None:
        return True

    return abs(candidate_duration - source_duration) <= VIDEO_DURATION_TOLERANCE_SECONDS


def source_media_matches(
    kind: str,
    duration: int | None,
    candidate_variants: tuple[ImageFingerprint, ...],
) -> bool:
    if not candidate_variants:
        return False

    for source in SOURCE_MEDIA:
        if source.kind != kind:
            continue

        if kind == "video" and not media_duration_compatible(duration, source.duration):
            continue

        if fingerprint_variants_match(candidate_variants, source.variants):
            return True

    return False


def add_source_media(
    kind: str,
    duration: int | None,
    variants: tuple[ImageFingerprint, ...],
) -> bool:
    global SOURCE_MEDIA, SOURCE_MEDIA_KEYS

    if not variants:
        return False

    primary = variants[0]
    key = (
        kind,
        duration if kind == "video" else None,
        primary.dhash,
        primary.ahash,
    )

    if key in SOURCE_MEDIA_KEYS:
        return False

    SOURCE_MEDIA.append(
        SourceMediaFingerprint(
            kind=kind,
            duration=duration,
            variants=variants,
        )
    )
    SOURCE_MEDIA_KEYS.add(key)

    if len(SOURCE_MEDIA) > SOURCE_MEDIA_CACHE_LIMIT:
        SOURCE_MEDIA = SOURCE_MEDIA[-SOURCE_MEDIA_CACHE_LIMIT:]
        SOURCE_MEDIA_KEYS = {
            (
                item.kind,
                item.duration if item.kind == "video" else None,
                item.variants[0].dhash,
                item.variants[0].ahash,
            )
            for item in SOURCE_MEDIA
            if item.variants
        }

    return True


# ---------------------------------------------------------------------------
# TELETHON SOURCE READER
# ---------------------------------------------------------------------------

def telethon_media_kind(message) -> tuple[str | None, int | None]:
    # Обычная Telegram photo.
    if getattr(message, "photo", None):
        return "photo", None

    file_info = getattr(message, "file", None)
    if file_info is None:
        return None, None

    mime_type = (getattr(file_info, "mime_type", None) or "").lower()
    duration = getattr(file_info, "duration", None)

    # Video, video note, GIF-like MP4 и video documents.
    if mime_type.startswith("video/") or getattr(file_info, "video_note", None):
        try:
            duration = int(duration) if duration is not None else None
        except (TypeError, ValueError):
            duration = None

        return "video", duration

    # Изображение, отправленное как document.
    if mime_type.startswith("image/"):
        return "photo", None

    return None, None


async def download_telethon_media_preview(message, kind: str) -> bytes | None:
    """
    Для видео скачиваем только Telegram thumbnail, а не весь ролик.
    Для фото сначала пробуем thumbnail; если не получилось — само фото.
    """

    # Пробуем несколько thumbnail indices, потому что самый большой thumb
    # иногда сам является video-size, а нам нужна картинка.
    for thumb in TELETHON_THUMB_CANDIDATES:
        try:
            data = await message.client.download_media(
                message,
                file=bytes,
                thumb=thumb,
            )
        except Exception:
            data = None

        if not data:
            continue

        # Проверяем, что это действительно изображение.
        variants = await asyncio.to_thread(fingerprint_image_bytes, data)
        if variants:
            return data

    # Для фото безопасно скачать само изображение целиком.
    # Для видео целиком файл не качаем — это слишком тяжело для Railway.
    if kind == "photo":
        try:
            data = await message.client.download_media(message, file=bytes)
            if data:
                return data
        except Exception:
            pass

    return None


async def process_source_message(message) -> tuple[int, int]:
    text_added = 1 if add_source_text(getattr(message, "message", None)) else 0
    media_added = 0

    kind, duration = telethon_media_kind(message)
    if kind is None:
        return text_added, media_added

    try:
        data = await download_telethon_media_preview(message, kind)
        if not data:
            return text_added, media_added

        variants = await asyncio.to_thread(fingerprint_image_bytes, data)
        if add_source_media(kind, duration, variants):
            media_added = 1

    except FloodWaitError as exc:
        logging.warning("Telethon flood wait: %s seconds", exc.seconds)
        await asyncio.sleep(exc.seconds)
    except Exception as exc:
        logging.debug(
            "Could not process source media message=%s: %s",
            getattr(message, "id", None),
            exc,
        )

    return text_added, media_added


async def scan_recent_source(limit: int) -> None:
    global SOURCE_LAST_ERROR

    if TARGET_ENTITY is None:
        return

    try:
        messages = await telethon_client.get_messages(TARGET_ENTITY, limit=limit)

        text_added = 0
        media_added = 0

        for message in reversed(messages):
            t, m = await process_source_message(message)
            text_added += t
            media_added += m

        SOURCE_LAST_ERROR = None
        logging.info(
            "Recent source scan complete: +texts=%s +media=%s totals=%s/%s",
            text_added,
            media_added,
            len(SOURCE_TEXTS),
            len(SOURCE_MEDIA),
        )

    except Exception as exc:
        SOURCE_LAST_ERROR = f"{type(exc).__name__}: {exc}"
        logging.warning("Recent source scan failed: %s", SOURCE_LAST_ERROR)


async def full_history_scan() -> None:
    global HISTORY_SCAN_RUNNING
    global HISTORY_SCAN_DONE
    global HISTORY_SCAN_PROGRESS
    global SOURCE_LAST_ERROR

    if TARGET_ENTITY is None or HISTORY_SCAN_RUNNING:
        return

    HISTORY_SCAN_RUNNING = True
    HISTORY_SCAN_DONE = False
    HISTORY_SCAN_PROGRESS = 0

    text_added = 0
    media_added = 0

    try:
        async for message in telethon_client.iter_messages(
            TARGET_ENTITY,
            limit=SOURCE_HISTORY_LIMIT,
            reverse=False,
        ):
            HISTORY_SCAN_PROGRESS += 1

            t, m = await process_source_message(message)
            text_added += t
            media_added += m

            if HISTORY_SCAN_PROGRESS % 100 == 0:
                logging.info(
                    "History scan progress=%s/%s texts=%s media=%s",
                    HISTORY_SCAN_PROGRESS,
                    SOURCE_HISTORY_LIMIT,
                    len(SOURCE_TEXTS),
                    len(SOURCE_MEDIA),
                )

            # Небольшая пауза снижает вероятность flood-limit
            # при большом количестве thumbnail downloads.
            if m:
                await asyncio.sleep(0.03)

        HISTORY_SCAN_DONE = True
        SOURCE_LAST_ERROR = None

        logging.info(
            "Full history scan complete: processed=%s +texts=%s +media=%s totals=%s/%s",
            HISTORY_SCAN_PROGRESS,
            text_added,
            media_added,
            len(SOURCE_TEXTS),
            len(SOURCE_MEDIA),
        )

    except FloodWaitError as exc:
        SOURCE_LAST_ERROR = f"FloodWait {exc.seconds}s"
        logging.warning("Full scan flood wait: %s seconds", exc.seconds)
        await asyncio.sleep(exc.seconds)

    except Exception as exc:
        SOURCE_LAST_ERROR = f"{type(exc).__name__}: {exc}"
        logging.exception("Full history scan failed")

    finally:
        HISTORY_SCAN_RUNNING = False


async def source_refresh_loop() -> None:
    while True:
        await asyncio.sleep(SOURCE_REFRESH_SECONDS)
        await scan_recent_source(SOURCE_REFRESH_LIMIT)


async def init_telethon() -> None:
    global TARGET_ENTITY
    global TARGET_CHANNEL_ID
    global TELETHON_CONNECTED
    global SOURCE_LAST_ERROR

    await telethon_client.connect()

    if not await telethon_client.is_user_authorized():
        raise RuntimeError(
            "TELETHON_SESSION не авторизован. Создай новую StringSession локально."
        )

    TELETHON_CONNECTED = True

    TARGET_ENTITY = await telethon_client.get_entity(
        f"@{TARGET_CHANNEL_USERNAME}"
    )

    # Telethon get_peer_id(add_mark=True) выдаёт Bot API-style channel id.
    TARGET_CHANNEL_ID = await telethon_client.get_peer_id(
        TARGET_ENTITY,
        add_mark=True,
    )

    SOURCE_LAST_ERROR = None

    me = await telethon_client.get_me()

    logging.info(
        "Telethon connected as user_id=%s. Target @%s -> %s",
        getattr(me, "id", None),
        TARGET_CHANNEL_USERNAME,
        TARGET_CHANNEL_ID,
    )


# ---------------------------------------------------------------------------
# INCOMING GROUP MEDIA (AIOGRAM)
# ---------------------------------------------------------------------------

def incoming_media_info(message: Message):
    """
    Возвращает:
      (kind, duration, downloadable object)
    """

    if message.photo:
        return "photo", None, message.photo[-1]

    if message.video_note:
        media = message.video_note.thumbnail
        if media:
            return "video", message.video_note.duration, media

    if message.video:
        media = message.video.thumbnail

        if media is None and message.video.cover:
            media = message.video.cover[-1]

        if media:
            return "video", message.video.duration, media

    if message.animation and message.animation.thumbnail:
        return "video", message.animation.duration, message.animation.thumbnail

    if message.document:
        mime_type = (message.document.mime_type or "").lower()

        if mime_type.startswith("image/"):
            # Если файл не слишком большой, Bot API сможет скачать само изображение.
            if (
                message.document.file_size is None
                or message.document.file_size <= 20 * 1024 * 1024
            ):
                return "photo", None, message.document

        if mime_type.startswith("video/") and message.document.thumbnail:
            return "video", None, message.document.thumbnail

        if message.document.thumbnail:
            return "photo", None, message.document.thumbnail

    return None, None, None


async def incoming_media_fingerprint(
    message: Message,
    bot: Bot,
) -> tuple[str | None, int | None, tuple[ImageFingerprint, ...]]:
    kind, duration, media = incoming_media_info(message)
    if media is None:
        return None, None, ()

    try:
        buffer = await bot.download(media)
        if buffer is None:
            return kind, duration, ()

        data = buffer.getvalue()
        variants = await asyncio.to_thread(fingerprint_image_bytes, data)

        return kind, duration, variants

    except Exception as exc:
        logging.warning(
            "Could not fingerprint incoming media message=%s: %s",
            message.message_id,
            exc,
        )
        return kind, duration, ()


# ---------------------------------------------------------------------------
# MODERATION
# ---------------------------------------------------------------------------

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
        logging.error(
            "No permission to delete in group %s",
            message.chat.id,
        )

    except TelegramBadRequest as exc:
        logging.warning("Telegram rejected deletion: %s", exc)


async def require_group_admin(message: Message, bot: Bot) -> bool:
    if message.chat.type not in GROUP_TYPES:
        await message.answer("Команда работает только внутри группы.")
        return False

    if message.from_user is None:
        await message.answer("Не удалось определить пользователя.")
        return False

    try:
        member = await bot.get_chat_member(
            message.chat.id,
            message.from_user.id,
        )
    except (TelegramBadRequest, TelegramForbiddenError):
        await message.answer("Не удалось проверить права администратора.")
        return False

    if member.status not in ADMIN_STATUSES:
        await message.answer("Команда доступна только администраторам группы.")
        return False

    return True


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Channel Guard v4\n\n"
        f"Источник: @{TARGET_CHANNEL_USERNAME}\n\n"
        "Удаляю:\n"
        "• настоящие Forward из источника;\n"
        "• только заданные связанные @username/t.me ссылки;\n"
        "• точные и почти точные копии текста;\n"
        "• скачанные и повторно загруженные фото;\n"
        "• обычные видео и видеокружки по Telegram thumbnail.\n\n"
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
    can_delete = (
        bool(getattr(member, "can_delete_messages", False))
        if is_admin
        else False
    )

    scan_state = (
        "идёт"
        if HISTORY_SCAN_RUNNING
        else ("завершён" if HISTORY_SCAN_DONE else "не запущен")
    )

    text = (
        "Статус Channel Guard v4:\n"
        f"Бот администратор: {'да' if is_admin else 'нет'}\n"
        f"Может удалять: {'да' if can_delete else 'нет'}\n"
        f"Telethon: {'подключён' if TELETHON_CONNECTED else 'не подключён'}\n"
        f"Источник: @{TARGET_CHANNEL_USERNAME}\n"
        f"Channel ID: {TARGET_CHANNEL_ID}\n"
        f"Кэш текста: {len(SOURCE_TEXTS)}\n"
        f"Кэш медиа: {len(SOURCE_MEDIA)}\n"
        f"Полный скан: {scan_state}\n"
        f"Обработано истории: {HISTORY_SCAN_PROGRESS}/{SOURCE_HISTORY_LIMIT}"
    )

    if SOURCE_LAST_ERROR:
        text += f"\nПоследняя ошибка: {SOURCE_LAST_ERROR}"

    text += (
        "\n\nМодерация готова."
        if is_admin and can_delete and TELETHON_CONNECTED
        else "\n\nПроверь права/переменные Railway."
    )

    await message.answer(text)


@router.message(Command("rescan"))
async def cmd_rescan(message: Message, bot: Bot) -> None:
    if not await require_group_admin(message, bot):
        return

    if HISTORY_SCAN_RUNNING:
        await message.answer(
            f"Полный скан уже идёт: {HISTORY_SCAN_PROGRESS}/{SOURCE_HISTORY_LIMIT}"
        )
        return

    asyncio.create_task(full_history_scan())
    await message.answer(
        f"Повторный скан последних {SOURCE_HISTORY_LIMIT} сообщений запущен."
    )


@router.message()
async def moderate_messages(message: Message, bot: Bot) -> None:
    if message.chat.type not in GROUP_TYPES:
        return

    # 1. Настоящий Forward из A.
    channel = forwarded_channel(message)

    if channel is not None:
        blocked_forward = (
            TARGET_CHANNEL_ID is not None
            and channel.id == TARGET_CHANNEL_ID
        )

        if not blocked_forward and channel.username:
            blocked_forward = (
                channel.username.casefold()
                == TARGET_CHANNEL_USERNAME.casefold()
            )

        if blocked_forward:
            await delete_message_safely(
                message,
                bot,
                "forward from blocked channel",
            )
            return

    # 2. Только конкретные связанные usernames/links.
    if contains_blocked_reference(message):
        await delete_message_safely(
            message,
            bot,
            "blocked Telegram username/link",
        )
        return

    # 3. Копия текста/подписи.
    copied, reason = is_source_text_copy(message_text(message))

    if copied:
        await delete_message_safely(message, bot, reason)
        return

    # 4. Повторно загруженное фото/видео/video note.
    if SOURCE_MEDIA:
        kind, duration, variants = await incoming_media_fingerprint(
            message,
            bot,
        )

        if (
            kind is not None
            and variants
            and source_media_matches(kind, duration, variants)
        ):
            await delete_message_safely(
                message,
                bot,
                "source media visual match",
            )
            return


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def validate_env() -> int:
    missing = []

    if not BOT_TOKEN:
        missing.append("BOT_TOKEN")

    if not API_ID_RAW:
        missing.append("API_ID")

    if not API_HASH:
        missing.append("API_HASH")

    if not TELETHON_SESSION:
        missing.append("TELETHON_SESSION")

    if missing:
        raise RuntimeError(
            "Не заданы Railway Variables: " + ", ".join(missing)
        )

    try:
        return int(API_ID_RAW)
    except ValueError as exc:
        raise RuntimeError("API_ID должен быть числом.") from exc


API_ID = validate_env()

telethon_client = TelegramClient(
    StringSession(TELETHON_SESSION),
    API_ID,
    API_HASH,
)


async def main() -> None:
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
            BotCommand(command="rescan", description="Пересканировать источник"),
            BotCommand(command="help", description="Инструкция"),
        ]
    )

    # 1. Подключаем user-account через Telethon.
    await init_telethon()

    # 2. Быстро загружаем последние публикации, чтобы бот сразу был полезен.
    await scan_recent_source(SOURCE_QUICK_SCAN_LIMIT)

    # 3. Полная история и регулярное обновление идут уже в background.
    history_task = asyncio.create_task(full_history_scan())
    refresh_task = asyncio.create_task(source_refresh_loop())

    logging.info(
        "Channel Guard v4 started. source=@%s quick_cache texts=%s media=%s",
        TARGET_CHANNEL_USERNAME,
        len(SOURCE_TEXTS),
        len(SOURCE_MEDIA),
    )

    try:
        await dp.start_polling(bot)

    finally:
        for task in (history_task, refresh_task):
            task.cancel()

        for task in (history_task, refresh_task):
            try:
                await task
            except asyncio.CancelledError:
                pass

        await telethon_client.disconnect()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
