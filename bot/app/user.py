import requests
from aiogram import Router, F, Bot
from aiogram.types import Message
from aiogram.filters import CommandStart
import app.keyboards as kb
import app.config as config  # Убедитесь, что импортировали config

TOKEN = config.TOKEN  # Получите токен из config
UPLOAD_API = 'http://94.241.141.190:9000/api/upload_user'

user = Router()

# Initialize the bot with the token from config
bot = Bot(token=TOKEN)

@user.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        f"""
Hello, {message.from_user.full_name}!
Welcome to ApplyBot!
Use /help to get help!
Good luck!
""", reply_markup=kb.main
    )

@user.message(F.text == 'HELP')
async def help(message: Message):
    await message.answer(f'''
Hello, {message.from_user.full_name}!
How it is done: 
1. You start this bot.
2. You send a picture of you with the code.

You MUST follow the next rules:
1. The picture must contain ONLY YOUR FACE AND THE CODE, NOTHING ELSE (especially prints, text on your background, t-shirt, etc.)

If you have more questions or if you are having any issue, contact @admin
''')

@user.message(F.photo)
async def get_or_post_face(message: Message):
    photo_id = message.photo[-1].file_id
    file = await bot.get_file(photo_id)
    file_path = file.file_path

    # Download the photo from Telegram server
    file_url = f'https://api.telegram.org/file/bot{TOKEN}/{file_path}'
    photo_bytes = requests.get(file_url).content

    # Send the photo to your FastAPI server
    files = {'image': ('photo.jpg', photo_bytes, 'image/jpeg')}
    response = requests.post(UPLOAD_API, files=files)

    if response.status_code == 200:
        result = response.json()
        await message.reply(f"Face registered successfully! ")
    else:
        await message.reply("Either face exists in database, or the format of the image you sent os incorrect(mens thta you probably disobeyed the rules).")
