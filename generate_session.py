from telethon.sync import TelegramClient
from telethon.sessions import StringSession

api_id = int(input("API_ID: ").strip())
api_hash = input("API_HASH: ").strip()

print()
print("Telegram попросит номер телефона, код и при необходимости 2FA-пароль.")
print()

with TelegramClient(StringSession(), api_id, api_hash) as client:
    session = client.session.save()

print()
print("=" * 70)
print("TELETHON_SESSION:")
print(session)
print("=" * 70)
print()
print("НЕ отправляй эту строку другим людям и не коммить её в GitHub.")
