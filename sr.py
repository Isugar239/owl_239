import speech_recognition as sr 
import pyaudio
import wave
def Audio2Text(AudioFile):
    r = sr.Recognizer()
    try:
      response = r.recognize_google(AudioFile, language="ru-RU")
      print("You said '" + response + "'")

    except sr.UnknownValueError:
      print("Could not understand audio")
    except sr.RequestError as e:
      print("Error; {0}".format(e))
CHUNK = 1024 
FRT = pyaudio.paInt16 
CHAN = 1 
RT = 44100 
REC_SEC = 5 
OUTPUT = "output.wav"
p = pyaudio.PyAudio()
stream = p.open(format=FRT,channels=CHAN,rate=RT,input=True,frames_per_buffer=CHUNK) # открываем поток для записи
print("rec")
frames = [] # формируем выборку данных фреймов
for i in range(0, int(RT / CHUNK * REC_SEC)):
    data = stream.read(CHUNK)
    frames.append(data)
print("done")

stream.stop_stream() # останавливаем и закрываем поток 
stream.close()
p.terminate()
w = wave.open(OUTPUT, 'wb')
w.setnchannels(CHAN)
w.setsampwidth(p.get_sample_size(FRT))
w.setframerate(RT)
w.writeframes(b''.join(frames))
w.close()
sample = sr.WavFile('output.wav')
r = sr.Recognizer()
with sample as audio:
    content = r.record(audio)
print(Audio2Text(audio))
