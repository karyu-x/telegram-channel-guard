import asyncio
import logging
import os
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ChatMemberStatus, ChatType
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.filters import Command, CommandStart
from aiogram.types import BotCommand, Message, MessageOriginChannel
from dotenv import load_dotenv

from storage import Storage

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
DB_PATH = os.getenv("DB_PATH", str(BASE_DIR / "guard.sqlite3")).strip()

router = Router()
storage = Storage(DB_PATH)

GROUP_TYPES = {ChatType.GROUP, ChatType.SUPERGROUP}
ADMIN_STATUSES = {ChatMemberStatus.CREATOR, ChatMemberStatus.ADMINISTRATOR}


def forwarded_channel(message: Message):
    """Return source channel Chat for a real Telegram forward, otherwise None."""
    origin = message.forward_origin
    if isinstance(origin, MessageOriginChannel):
        return origin.chat
    return None


async def require_group_admin(message: Message, bot: Bot) -> bool:
    if message.chat.type not in GROUP_TYPES:
        await message.answer("Эта команда работает только внутри группы.")
        return False

    if message.from_user is None:
        await message.answer(
            "Не удалось определить пользователя. Отправь команду от своего аккаунта, "
            "а не анонимно от имени группы."
        )
        return False

    try:
        member = await bot.get_chat_member(message.chat.id, message.from_user.id)
    except (TelegramBadRequest, TelegramForbiddenError):
        await message.answer("Не удалось проверить права администратора.")
        return False

    if member.status not in ADMIN_STATUSES:
        await message.answer("Эта команда доступна только администраторам группы.")
        return False

    return True


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Я удаляю пересланные посты из запрещённых Telegram-каналов.\n\n"
        "Как заблокировать канал:\n"
        "1. Добавь меня в группу администратором.\n"
        "2. Дай право удалять сообщения.\n"
        "3. Перешли в группу любой пост нужного канала.\n"
        "4. Ответь на этот пост командой /blockchannel\n\n"
        "Команды:\n"
        "/blockchannel — заблокировать канал по пересланному посту\n"
        "/unblockchannel — разблокировать канал\n"
        "/blockedchannels — список блокировок\n"
        "/status — проверить права бота\n"
        "/help — инструкция"
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await cmd_start(message)


@router.message(Command("status"))
async def cmd_status(message: Message, bot: Bot) -> None:
    if message.chat.type not in GROUP_TYPES:
        await message.answer("Проверять статус нужно внутри группы.")
        return

    me = await bot.get_me()
    try:
        member = await bot.get_chat_member(message.chat.id, me.id)
    except (TelegramBadRequest, TelegramForbiddenError) as exc:
        await message.answer(f"Не удалось получить мои права: {exc}")
        return

    is_admin = member.status in ADMIN_STATUSES
    can_delete = bool(getattr(member, "can_delete_messages", False)) if is_admin else False

    await message.answer(
        "Статус бота:\n"
        f"Администратор: {'да' if is_admin else 'нет'}\n"
        f"Может удалять сообщения: {'да' if can_delete else 'нет'}\n\n"
        + (
            "Всё готово."
            if is_admin and can_delete
            else "Сделай бота администратором и включи право удаления сообщений."
        )
    )


@router.message(Command("blockchannel"))
async def cmd_block_channel(message: Message, bot: Bot) -> None:
    if not await require_group_admin(message, bot):
        return

    target = message.reply_to_message
    if target is None:
        await message.answer(
            "Ответь командой /blockchannel именно на пересланный пост из канала."
        )
        return

    channel = forwarded_channel(target)
    if channel is None:
        await message.answer(
            "У этого сообщения нет определяемого источника-канала.\n"
            "Нужна обычная пересылка через Forward, а не скопированный текст/файл."
        )
        return

    added = await storage.block_channel(
        group_id=message.chat.id,
        channel_id=channel.id,
        title=channel.title or "Без названия",
        username=channel.username,
        added_by=message.from_user.id if message.from_user else None,
    )

    label = f"@{channel.username}" if channel.username else str(channel.id)
    if added:
        await message.answer(
            f"Канал заблокирован: {channel.title or 'Без названия'} ({label}).\n"
            "Новые пересланные посты из него будут автоматически удаляться."
        )
    else:
        await message.answer(
            f"Этот канал уже заблокирован: {channel.title or 'Без названия'} ({label})."
        )


@router.message(Command("unblockchannel"))
async def cmd_unblock_channel(message: Message, bot: Bot) -> None:
    if not await require_group_admin(message, bot):
        return

    channel_id = None
    channel_title = None

    if message.reply_to_message:
        channel = forwarded_channel(message.reply_to_message)
        if channel:
            channel_id = channel.id
            channel_title = channel.title

    if channel_id is None and message.text:
        parts = message.text.split(maxsplit=1)
        if len(parts) == 2:
            try:
                channel_id = int(parts[1].strip())
            except ValueError:
                pass

    if channel_id is None:
        await message.answer(
            "Ответь /unblockchannel на пересланный пост из канала\n"
            "или используй:\n/unblockchannel -1001234567890"
        )
        return

    removed = await storage.unblock_channel(message.chat.id, channel_id)
    if removed:
        await message.answer(
            f"Канал разблокирован"
            + (f": {channel_title}" if channel_title else f": {channel_id}")
            + "."
        )
    else:
        await message.answer("Такого канала нет в списке блокировок этой группы.")


@router.message(Command("blockedchannels"))
async def cmd_blocked_channels(message: Message, bot: Bot) -> None:
    if not await require_group_admin(message, bot):
        return

    channels = await storage.list_blocked(message.chat.id)
    if not channels:
        await message.answer("В этой группе пока нет заблокированных каналов.")
        return

    lines = ["Заблокированные каналы:"]
    for index, item in enumerate(channels, start=1):
        username = f"@{item['username']}" if item["username"] else "без username"
        lines.append(
            f"{index}. {item['title']} — {username}\n"
            f"   ID: {item['channel_id']}"
        )

    await message.answer("\n".join(lines))


@router.message()
async def moderate_forwarded_posts(message: Message, bot: Bot) -> None:
    if message.chat.type not in GROUP_TYPES:
        return

    channel = forwarded_channel(message)
    if channel is None:
        return

    if not await storage.is_blocked(message.chat.id, channel.id):
        return

    try:
        await bot.delete_message(message.chat.id, message.message_id)
        logging.info(
            "Deleted blocked forward: group=%s channel=%s message=%s",
            message.chat.id,
            channel.id,
            message.message_id,
        )
    except TelegramForbiddenError:
        logging.error(
            "Cannot delete message in group %s: bot has insufficient rights",
            message.chat.id,
        )
    except TelegramBadRequest as exc:
        logging.warning("Telegram rejected deletion: %s", exc)


async def main() -> None:
    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN не задан. Скопируй .env.example в .env и вставь токен от @BotFather."
        )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    await storage.init()

    bot = Bot(BOT_TOKEN)
    dp = Dispatcher()
    dp.include_router(router)

    await bot.set_my_commands(
        [
            BotCommand(command="blockchannel", description="Заблокировать канал"),
            BotCommand(command="unblockchannel", description="Разблокировать канал"),
            BotCommand(command="blockedchannels", description="Список блокировок"),
            BotCommand(command="status", description="Проверить права бота"),
            BotCommand(command="help", description="Инструкция"),
        ]
    )

    try:
        logging.info("Channel Guard started")
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
