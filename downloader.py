from pyrogram import Client

API_ID = None # your telegram api id from my.telegram.org
API_HASH = None # your telegram api hash from my.telegram.org
BOT_TOKEN = None # your telegram bot token from @BotFather
CHAT_ID = # your gallery channel id
CHAT_START_ID = 1 # your gallery channel starting image id
CHAT_END_ID = 1000 # your gallery channel ending image id
proxy = None # your proxy settings, can be None

app = Client("gallery_downloader", api_id=API_ID, api_hash=API_HASH,
             bot_token=BOT_TOKEN, proxy=proxy)


file_unique_ids = set()
file_ids = {}
message_ids = {}

async def download_all():
    async with app:
        for i in range(CHAT_START_ID, CHAT_END_ID, 200):
            photos = await app.get_messages(CHAT_ID, [j for j in range(i, i+200)])
            for photo in photos:
                if photo.photo is not None:
                    file_unique_id = photo.photo.file_unique_id
                    file_id = photo.photo.file_id
                    message_id = photo.id
                    if file_unique_id in file_unique_ids:
                        continue
                    file_unique_ids.add(file_unique_id)
                    file_ids[file_unique_id] = file_id
                    message_ids[file_unique_id] = message_id
        for file_unique_id in file_ids:
            await app.download_media(file_ids[file_unique_id], file_name=f'data/{message_ids[file_unique_id]}.jpg')

app.run(download_all())