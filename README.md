# FaceGalleryTelegramBot
A Telegram bot to find your face among all the pictures in a gallery channel. it uses `Multi-Task Cascaded Convolutional Neural Networks` implemented in `facenet-pytorch`.
it also uses `pyrogram` for Telegram API. Credits to Mahdi Samiee and Iman Mohammadi for the idea.
## Installation
First we have to install `numpy`, `cv2`, and `torch`. then we can run the command:
`pip install facenet-pytorch pyrogram`
then we have to get a bot token from `@BotFather`. then we have to get `API ID` and `API Hash` from [here](my.telegram.org). lastly we must replace all the vaules in the files.
## Basic Usage
first we extract our gallery chat id using `@RawDataBot`. then we run `downloader`.
when all the photos are downloaded in `data` directory. run `indexer` notebook so the embeddings are saved in and `idx` file.
finally, add the index file in the `bot.py` and enjoy.
