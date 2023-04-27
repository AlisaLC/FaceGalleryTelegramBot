from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
from numpy.linalg import norm
import heapq
import cv2
from pyrogram import Client, filters
from pyrogram.types import InlineKeyboardMarkup, InlineKeyboardButton
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, select_largest=False, keep_all=True,
    device=device
)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

API_ID = None # your telegram api id from my.telegram.org
API_HASH = None # your telegram api hash from my.telegram.org
BOT_TOKEN = None # your telegram bot token from @BotFather
proxy = None # your proxy settings, can be None

app = Client("galley_bot", api_id=API_ID, api_hash=API_HASH,
             bot_token=BOT_TOKEN, proxy=proxy)

message_embeddings = {}

# add your indexes here
indexes = []
embedding_to_file = {}
for index in indexes:
    with open(index, 'rb') as f:
        embedding_to_file_partial = pickle.load(f)
        embedding_to_file.update(embedding_to_file_partial)


@app.on_callback_query()
async def continue_find(client, callback_query):
    key, end = callback_query.data.split('|')
    key, end = int(key), int(end)
    start = end - 10
    if key not in message_embeddings:
        await callback_query.answer('لینک منقضی شده. بار دیگر عکس را بفرستید.')
        return
    embeddings = message_embeddings[key]
    await callback_query.answer()
    for i, similar_image in enumerate(find_similars(embeddings)):
        if i < start:
            continue
        if i == end:
            break
        await client.forward_messages(callback_query.message.chat.id, similar_image[0], similar_image[1])
    await callback_query.message.edit_text(text='برای مشاهده تصاویر بیشتر بر روی دکمه زیر کلیک کنید.',
                                           reply_markup=InlineKeyboardMarkup([[
                                               InlineKeyboardButton('مشاهده تصاویر بیشتر ' + f'{end}/{end+10}',
                                                                    callback_data=f'{key}|{end+10}')
                                           ]]))


@app.on_message(filters.private & filters.photo)
async def start_find(client, message):
    photo = message.photo
    print(photo)
    image = await client.download_media(photo.file_id)
    i = 0
    try:
        embedding = get_embeddings(image)
        if len(embedding) == 0:
            raise Exception('No Face')
        embedding = embedding[0]
        key = int(np.random.randint(1_000_000_000))
        if message.media_group_id is None:
            message_embeddings[key] = [embedding]
            await client.send_message(chat_id=message.chat.id, text='برای مشاهده تصاویر بیشتر بر روی دکمه زیر کلیک کنید.',
                                      reply_markup=InlineKeyboardMarkup([[
                                          InlineKeyboardButton(
                                              'مشاهده تصاویر بیشتر 0/10', callback_data=f'{key}|10')
                                      ]]))
        else:
            key = int(message.media_group_id)
            if key in message_embeddings:
                message_embeddings[key].append(embedding)
            else:
                message_embeddings[key] = [embedding]
                await client.send_message(chat_id=message.chat.id, text='برای مشاهده تصاویر بیشتر بر روی دکمه زیر کلیک کنید.',
                                          reply_markup=InlineKeyboardMarkup([[
                                              InlineKeyboardButton(
                                                  'مشاهده تصاویر بیشتر 0/10', callback_data=f'{key}|10')
                                          ]]))
    except:
        await client.send_message(chat_id=message.chat.id, text='متاسفانه صورت شما در تصویر قابل تشخیص نبود.')


@app.on_message(filters.private & filters.command(["start", "help"]))
async def my_handler(client, message):
    await client.send_message(chat_id=message.chat.id, text='یه عکس از خودت بفرست تا عکسای رویدادتو برات بفرستم.')


def get_embeddings(image):
    image = cv2.imread(image)
    x_aligned = mtcnn(image)
    if x_aligned is not None:
        return resnet(x_aligned.to(device)).detach().cpu().numpy()
    return np.array([])


def cosine_similarity(A, B):
    return (A @ B) / (norm(A) * norm(B))


def find_similars(embeddings):
    similars = []
    for emb in embedding_to_file.keys():
        emb_np = np.array(list(emb))
        similars.append((-sum(cosine_similarity(embedding, emb_np)
                        for embedding in embeddings), emb))
    heapq.heapify(similars)
    image_set = set()
    while similars:
        _, emb = heapq.heappop(similars)
        image = embedding_to_file[emb]
        if image in image_set:
            continue
        image_set.add(image)
        yield image


app.run()
