import os
from typing import List
from TTS.api import TTS
from pydub import AudioSegment
from gtts import gTTS
# OUTPUT_DIR = "/media/olegg/sova/owl_239"

# tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

def generate_wait_prompts(phrases: List[str]) -> None:
    for idx, phrase in enumerate(phrases, start=1):
        # out_mp3 = os.path.join(OUTPUT_DIR, f"start.mp3")
        # tts.tts_to_file(text=phrase, file_path=out_mp3, speaker_wav="speaker.wav", language="ru")
        tts = gTTS(phrase, lang="ru")
        # Сохраняем сначала во временный MP3, затем конвертируем в WAV нужной частоты
        tts.save("/media/olegg/sova/owl_239/start.mp3")
        sound = AudioSegment.from_mp3("/media/olegg/sova/owl_239/start.mp3")
        sound.export("/media/olegg/sova/owl_239/start.wav", format="wav")

        

if __name__ == "__main__":
    phrases_ru = [
        "Здравствуйте, я сова -  интерактивный робот помощник. Надеюсь вам понравилось выступление моей подруги вороны, а сейчас можете задать мне вопросы."
    ]
    generate_wait_prompts(phrases_ru)


