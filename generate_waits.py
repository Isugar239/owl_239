import os
from typing import List

from TTS.api import TTS
from pydub import AudioSegment


OUTPUT_DIR = "/media/olegg/sova/owl_239"
SPEAKER_WAV = "/media/olegg/sova/owl_239/speaker.wav"


def save_mp3_from_wav(wav_path: str, mp3_path: str) -> None:
    audio = AudioSegment.from_wav(wav_path)
    audio.export(mp3_path, format="mp3")


def generate_wait_prompts(phrases: List[str]) -> None:
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
    for idx, phrase in enumerate(phrases, start=1):
        tmp_wav = os.path.join(OUTPUT_DIR, f"wait{idx}.wav")
        out_mp3 = os.path.join(OUTPUT_DIR, f"wait{idx}.mp3")
        tts.tts_to_file(
            text=phrase,
            file_path=tmp_wav,
            speaker_wav=SPEAKER_WAV,
            language="ru",
        )
        save_mp3_from_wav(tmp_wav, out_mp3)
        try:
            os.remove(tmp_wav)
        except OSError:
            pass


if __name__ == "__main__":
    phrases_ru = [
        "Секунду, я думаю...",
        "Пожалуйста, немного подождите.",
        "Обрабатываю ваш вопрос...",
        "Уже думаю над ответом.",
    ]
    generate_wait_prompts(phrases_ru)


