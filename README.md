# Telegram Channel Guard v2

Без базы данных.

Блокирует:
- Forward из `@ibragimmansurov_blog`;
- `@manager_ibragimmansurov`;
- `@ibragimmansurov_blog`;
- ссылки на `t.me/ibragimmansurov_blog`;
- скрытые Telegram `text_link`, ведущие на запрещённые username;
- точные и почти точные текстовые копии последних постов источника.

Для сравнения бот каждые 30 секунд читает публичную preview-страницу
`https://t.me/s/ibragimmansurov_blog` и хранит тексты только в RAM.

## Railway

Variables:

```env
BOT_TOKEN=...
```

Start Command:

```bash
python bot.py
```

Volume и DB_PATH не нужны.

После deploy выполни в группе:

```text
/status
```

Статус покажет, сколько исходных текстов загружено в кэш.

## Ограничения

Копия фото/видео без подписи пока не определяется. Сильно переписанный текст
тоже намеренно не удаляется, чтобы не ловить обычные похожие сообщения.
