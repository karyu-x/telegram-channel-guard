import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, Router
from aiogram.enums import ChatMemberStatus, ChatType
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.filters import Command, CommandStart
from aiogram.types import BotCommand, Message, MessageOriginChannel
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

# Единственный канал, пересылки из которого запрещены.
TARGET_CHANNEL_USERNAME = "ibragimmansurov_blog"

router = Router()

GROUP_TYPES = {ChatType.GROUP, ChatType.SUPERGROUP}
ADMIN_STATUSES = {ChatMemberStatus.CREATOR, ChatMemberStatus.ADMINISTRATOR}

# ID канала определяется при запуске через @username и хранится только в памяти.
TARGET_CHANNEL_ID: int | None = None


def forwarded_channel(message: Message):
    origin = message.forward_origin
    if isinstance(origin, MessageOriginChannel):
        return origin.chat
    return None


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Я автоматически удаляю пересланные посты из "
        f"@{TARGET_CHANNEL_USERNAME}.\n\n"
        "Добавь меня в группу администратором и дай право удалять сообщения.\n"
        "Команда /status проверит, всё ли настроено."
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
        f"Канал @{TARGET_CHANNEL_USERNAME} найден, ID: {TARGET_CHANNEL_ID}"
        if TARGET_CHANNEL_ID is not None
        else f"Не удалось определить ID @{TARGET_CHANNEL_USERNAME}"
    )

    await message.answer(
        "Статус:\n"
        f"Администратор: {'да' if is_admin else 'нет'}\n"
        f"Может удалять сообщения: {'да' if can_delete else 'нет'}\n"
        f"{target_status}\n\n"
        + (
            "Всё готово."
            if is_admin and can_delete and TARGET_CHANNEL_ID is not None
            else "Проверь права бота и логи Railway."
        )
    )


@router.message()
async def moderate_forwarded_posts(message: Message, bot: Bot) -> None:
    if message.chat.type not in GROUP_TYPES:
        return

    channel = forwarded_channel(message)
    if channel is None:
        return

    # Основная проверка по постоянному Telegram ID.
    blocked = TARGET_CHANNEL_ID is not None and channel.id == TARGET_CHANNEL_ID

    # Резервная проверка по username на случай, если ID не удалось получить при старте.
    if not blocked and channel.username:
        blocked = channel.username.lower() == TARGET_CHANNEL_USERNAME.lower()

    if not blocked:
        return

    try:
        await bot.delete_message(
            chat_id=message.chat.id,
            message_id=message.message_id,
        )
        logging.info(
            "Deleted blocked forward: group=%s channel=%s username=%s message=%s",
            message.chat.id,
            channel.id,
            channel.username,
            message.message_id,
        )
    except TelegramForbiddenError:
        logging.error(
            "Не хватает прав для удаления сообщений в группе %s",
            message.chat.id,
        )
    except TelegramBadRequest as exc:
        logging.warning("Telegram отклонил удаление сообщения: %s", exc)


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
            "Не удалось определить ID @%s: %s. "
            "Будет использоваться резервная проверка по username.",
            TARGET_CHANNEL_USERNAME,
            exc,
        )


async def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN не задан. Добавь BOT_TOKEN в Railway Variables или файл .env."
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
            BotCommand(command="status", description="Проверить права бота"),
            BotCommand(command="help", description="Инструкция"),
        ]
    )

    await resolve_target_channel(bot)

    try:
        logging.info(
            "Channel Guard started. Blocking forwards from @%s",
            TARGET_CHANNEL_USERNAME,
        )
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
