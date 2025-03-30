from TTS.api import TTS
import os
file_path="output.wav"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.to("cuda:0")
tts.tts_to_file(text="В центре Санкт-петербурга находится лицей номер 239. Возраст лицея, а в прошлом школы номер 239, более 80 лет. ",
                file_path=file_path,
                speaker_wav='speaker.wav',
            language="ru")
