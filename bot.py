import asyncio
import logging
import os
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Iterable

import aiohttp
from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ChatMemberStatus, ChatType
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.filters import Command, CommandStart
from aiogram.types import BotCommand, Message, MessageEntity, MessageOriginChannel
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

TARGET_CHANNEL_USERNAME = "ibragimmansurov_blog"
TARGET_CHANNEL_PREVIEW = f"https://t.me/s/{TARGET_CHANNEL_USERNAME}"

BANNED_USERNAMES = {
    "manager_ibragimmansurov",
    "ibragimmansurov_blog",
}

SOURCE_BOOTSTRAP_PAGES = 5
SOURCE_CACHE_LIMIT = 500
SOURCE_REFRESH_SECONDS = 30

FUZZY_MIN_LENGTH = 80
FUZZY_THRESHOLD = 0.94

router = Router()
GROUP_TYPES = {ChatType.GROUP, ChatType.SUPERGROUP}
ADMIN_STATUSES = {ChatMemberStatus.CREATOR, ChatMemberStatus.ADMINISTRATOR}

TARGET_CHANNEL_ID: int | None = None
SOURCE_TEXTS: list[str] = []
SOURCE_TEXT_SET: set[str] = set()
SOURCE_LAST_ERROR: str | None = None

ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
WS_RE = re.compile(r"\s+")

URL_USERNAME_RE = re.compile(
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


def contains_banned_reference(message: Message) -> bool:
    normalized = normalize_text(message_text(message))

    for username in BANNED_USERNAMES:
        if f"@{username.casefold()}" in normalized:
            return True

    for match in URL_USERNAME_RE.finditer(normalized):
        if match.group("username").casefold() in BANNED_USERNAMES:
            return True

    for entity in message_entities(message):
        if entity.url:
            url = normalize_text(entity.url)
            for username in BANNED_USERNAMES:
                if username.casefold() in url:
                    return True

    return False


def is_source_copy(text: str) -> tuple[bool, str]:
    candidate = normalize_text(text)
    if not candidate:
        return False, ""

    if candidate in SOURCE_TEXT_SET:
        return True, "exact source text"

    if len(candidate) < FUZZY_MIN_LENGTH:
        return False, ""

    for source in SOURCE_TEXTS:
        if len(source) < FUZZY_MIN_LENGTH:
            continue

        shorter = min(len(candidate), len(source))
        longer = max(len(candidate), len(source))

        if shorter / longer >= 0.82 and (candidate in source or source in candidate):
            return True, "source text containment"

    for source in SOURCE_TEXTS:
        if len(source) < FUZZY_MIN_LENGTH:
            continue

        length_ratio = min(len(candidate), len(source)) / max(len(candidate), len(source))
        if length_ratio < 0.78:
            continue

        ratio = SequenceMatcher(None, candidate, source, autojunk=False).ratio()
        if ratio >= FUZZY_THRESHOLD:
            return True, f"source similarity={ratio:.3f}"

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

    if len(SOURCE_TEXTS) > SOURCE_CACHE_LIMIT:
        SOURCE_TEXTS = SOURCE_TEXTS[-SOURCE_CACHE_LIMIT:]
        SOURCE_TEXT_SET = set(SOURCE_TEXTS)

    return added


async def fetch_preview_page(
    session: aiohttp.ClientSession,
    before: int | None = None,
) -> tuple[list[str], int | None]:
    url = TARGET_CHANNEL_PREVIEW
    if before is not None:
        url = f"{url}?before={before}"

    async with session.get(url, allow_redirects=True) as response:
        response.raise_for_status()
        html = await response.text()

    soup = BeautifulSoup(html, "html.parser")

    texts: list[str] = []
    ids: list[int] = []

    for wrap in soup.select(".tgme_widget_message_wrap"):
        message = wrap.select_one(".tgme_widget_message")
        if message:
            data_post = message.get("data-post")
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

    return texts, min(ids) if ids else None


async def bootstrap_source_cache() -> None:
    global SOURCE_LAST_ERROR

    timeout = aiohttp.ClientTimeout(total=15)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/151 Safari/537.36"
        )
    }

    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            before = None
            all_texts: list[str] = []

            for _ in range(SOURCE_BOOTSTRAP_PAGES):
                texts, next_before = await fetch_preview_page(session, before)
                all_texts.extend(texts)

                if next_before is None or next_before == before:
                    break
                before = next_before

            added = add_source_texts(all_texts)
            SOURCE_LAST_ERROR = None
            logging.info(
                "Source cache bootstrapped: total=%s newly_added=%s",
                len(SOURCE_TEXTS),
                added,
            )
    except Exception as exc:
        SOURCE_LAST_ERROR = f"{type(exc).__name__}: {exc}"
        logging.warning("Source bootstrap failed: %s", SOURCE_LAST_ERROR)


async def refresh_source_cache_once() -> None:
    global SOURCE_LAST_ERROR

    timeout = aiohttp.ClientTimeout(total=15)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/151 Safari/537.36"
        )
    }

    try:
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            texts, _ = await fetch_preview_page(session)
            added = add_source_texts(texts)

        SOURCE_LAST_ERROR = None
        if added:
            logging.info("Source cache refreshed: +%s total=%s", added, len(SOURCE_TEXTS))
    except Exception as exc:
        SOURCE_LAST_ERROR = f"{type(exc).__name__}: {exc}"
        logging.warning("Source refresh failed: %s", SOURCE_LAST_ERROR)


async def source_refresh_loop() -> None:
    while True:
        await asyncio.sleep(SOURCE_REFRESH_SECONDS)
        await refresh_source_cache_once()


async def delete_message_safely(message: Message, bot: Bot, reason: str) -> None:
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
        "Я удаляю:\n"
        f"1. пересылки из @{TARGET_CHANNEL_USERNAME};\n"
        "2. сообщения со ссылками/упоминаниями запрещённых аккаунтов;\n"
        "3. текстовые копии и почти точные копии последних постов источника.\n\n"
        "Для работы в группе дай мне права администратора на удаление сообщений.\n"
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
        f"Кэш постов: {len(SOURCE_TEXTS)} текстов"
        if SOURCE_TEXTS
        else "Кэш постов: пуст"
    )

    if SOURCE_LAST_ERROR:
        cache_status += f"\nОшибка обновления источника: {SOURCE_LAST_ERROR}"

    await message.answer(
        "Статус:\n"
        f"Администратор: {'да' if is_admin else 'нет'}\n"
        f"Может удалять сообщения: {'да' if can_delete else 'нет'}\n"
        f"{target_status}\n"
        f"{cache_status}\n\n"
        + (
            "Основная модерация готова."
            if is_admin and can_delete
            else "Сначала выдай боту право удаления сообщений."
        )
    )


@router.message()
async def moderate_messages(message: Message, bot: Bot) -> None:
    if message.chat.type not in GROUP_TYPES:
        return

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

    if contains_banned_reference(message):
        await delete_message_safely(message, bot, "banned username/link")
        return

    copied, reason = is_source_copy(message_text(message))
    if copied:
        await delete_message_safely(message, bot, reason)


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
        raise RuntimeError("BOT_TOKEN не задан. Добавь BOT_TOKEN в Railway Variables.")

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
            "Channel Guard v2 started. Blocking @%s. Cached source texts=%s",
            TARGET_CHANNEL_USERNAME,
            len(SOURCE_TEXTS),
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
