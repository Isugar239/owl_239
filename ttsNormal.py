from TTS.api import TTS
import os
file_path="speaker.wav"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.to("cuda:0")
tts.tts_to_file(text="Слушаю Вас ",
                file_path="listen.wav",
                speaker_wav=file_path,
            language="ru")
