from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="Where is the delivery robot?",
                file_path="output1.wav",
                speaker_wav="onstage.wav",
                language="en")
tts.tts_to_file(text="Oh oh oh!!",
                file_path="output2.wav",
                speaker_wav="onstage.wav",
                language="en")
tts.tts_to_file(text="Give me an orange!",
                file_path="output3.wav",
                speaker_wav="onstage.wav",
                language="en")
tts.tts_to_file(text="I want an orange. Give me an orange!",
                file_path="output4.wav",
                speaker_wav="onstage.wav",
                language="en")
tts.tts_to_file(text="Oh no, my friend Gena the crocodile has fallen!",
                file_path="output5.wav",
                speaker_wav="onstage.wav",
                language="en")