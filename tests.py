from transformers import pipeline
import soundfile as sf

pipe = pipeline("text-to-speech", model="Misha24-10/F5-TTS_RUSSIAN")
print("init")
result = pipe("я сова которой 100 лет")

print(result)

sf.write("test.wav", result["audio"].squeeze(), result["sampling_rate"])

print("end")