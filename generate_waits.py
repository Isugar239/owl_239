import os
from typing import List

from gtts import gTTS
from gtts import gTTS


OUTPUT_DIR = "/media/olegg/sova/owl_239"


def generate_wait_prompts(phrases: List[str]) -> None:
    for idx, phrase in enumerate(phrases, start=1):
        out_mp3 = os.path.join(OUTPUT_DIR, f"wait{idx}.mp3")
        tts = gTTS(text=phrase, lang="ru")
        tts.save(out_mp3)


if __name__ == "__main__":
    phrases_ru = [
        "Секунду, я думаю...",
        "Пожалуйста, немного подождите.",
        "Обрабатываю ваш вопрос...",
        "Уже думаю над ответом.",
    ]
    generate_wait_prompts(phrases_ru)


