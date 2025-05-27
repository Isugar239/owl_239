import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSpeechSeq2Seq, AutoProcessor, BertTokenizerFast, AutoModelForCausalLM, TFBertModel
import time
import sounddevice as sd
import numpy as np
import wave
from TTS.api import TTS
import pygame
import random
import serial
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf
from types import SimpleNamespace
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from pydub import AudioSegment
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#init apc220

HAVE_CAM = 1 # отладка с камерой \ без
HAVE_BT = 1 # отладка с блютузом \ без
HAVE_APC = 1 # отладка с apc serial
last_phrase = 0
if HAVE_APC:
    port = '/dev/ttyUSB0'
    baud_rate = 9600
    timeout = 1
    ser = serial.Serial(port, baud_rate, timeout=timeout)
# else:
#     ser = SimpleNamespace()
#     ser.write(*args) = lambda self: "No apc"
pygame.init()
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
def contains_negative_words(text):
    negative_words = ["нет", "не", "никак", "вряд ли", "никогда", "ни за что", "неа", "отрицательно", "исключено", "не думаю", "сомневаюсь"]
    text_lower = text.lower()
    return any(neg in text_lower for neg in negative_words)

mp_face_detection = mp.solutions.face_detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

gesture_model = load_model('mp_hand_gesture')

classNames = ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']

torch.random.manual_seed(0)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
def record_audio(filename="voice.wav", duration=5, samplerate=44100, device_name=None):
    print("Доступные устройства:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']}")
    
    device_id = None
    if device_name:
        for i, device in enumerate(devices):
            if device_name.lower() in device['name'].lower():
                device_id = i
                print(f"Выбрано устройство: {device['name']}")
                break
    
    if device_id is None:
        print("Устройство не найдено, используется устройство по умолчанию")
    
    print("rec start")
    audio_data = sd.rec(
        int(samplerate * duration),
        samplerate=samplerate,
        channels=1,
        dtype=np.int16,
        device=device_id
    )
    sd.wait()
    print("rec end")
    
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())
    return filename

context = '''Я выступала на многих соревнованиях в числе которых Робокап 2025
 Президентский Физико математический лицей 239 распологается в центре санкт петербурга между фруштатской и кирочной улиц. Там я нахожусь прямо сейчас. Школа была основана в 1918 году, на данный момент ей 107 лет.
Директор лицея - Максим Яковлевич Пратусеевич. Там преподают такие предметы как: Алгебра, геометрия, иностранный язык, информатика, история, литература, русский язык, физика, Химия, биология, география, физкультура, искусство, обществознание, ОБЖ.
В лицее есть классы с 5 по 11, по несколько паралелей в каждом. Есть химбио классы и еще по многим направлениям. В конце года есть переводные экзамены. Средней бал ЕГЭ у выпускников по всем предметам около 60 баллов.
В 239 много кружков на самые разные направления, включая матцентр, физцентр, робоцентр и т п, суммарно около 100. Поэтому в среднем 30-40 человек выпускаются с возможностью поступления БВИ.
Периодами промежуточной аттестации в V–IX классах являются четверти, а в X–XI классах — полугодия. 
К основным видам промежуточной аттестации в ФМЛ № 239 относятся:  экзамен зачёт контрольная работа интегрированный проект с использованием ИКТ  сочинение изложение диктант с грамматическим заданием тестовая работа собеседование исследовательская работа
Экзамены чаще всего в мае-июне, аттестации в конце года в декабре-январе.
Праздник Последнего звонка — всегда грустный праздник. Ведь приходится расставаться со школой, с её удивительным миром. И, уходя, каждый 11 класс прощается со школой: выходит на сцену, чтобы ещё раз сказать спасибо за все. 
Форма была введена в 2007 году. А родилась идея ввести форму после того, как на одном из очередных награждений за победы в городских олимпиадах на сцену вышла команда 239: в рваных футболках, страшных джинсах, лохматые, непричёсанные. 
А в 2012 году была придумана ещё одна традиция 239 — Золотой директорский галстук. Его можно получить как высшую награду за отличную учёбу.
 '''

messages = [
    {"role": "system", "content": f"Ты сова, которая отвечает на вопросы по слеудющему тексту \n{context}\n. Если не знаешь - не пытайся угадать, признайся что не знаешь. Все цифры заменяй словами в нужном падеже. В конце ответа не ставь точку. Если в вопросе есть слово похожее на лицей, считай что это оно. основываясь ТОЛЬКО на данных, переданных в запросе дай только ответ ТОЛЬКО на этот вопрос"},
    {"role": "user", "content": "Кто директор 239? В конце ответа не ставь точку"},
    {"role": "assistant", "content": "Максим Яковлевич Пратусеевич"},       
    {"role": "user", "content": "Сколько человек в 11 классе?"},
    {"role": "assistant", "content": "Данной информации у меня нет"},       
]

model_path = "microsoft/Phi-4-mini-instruct"
generation_args = {
    "max_new_tokens": 512,
    "return_full_text": False,
    "do_sample": False,
}
file_path="speaker.wav"

def addUwu(pathToSound):
    song = AudioSegment.from_wav(pathToSound)
    song2 = AudioSegment.from_wav("UWU.wav")
    song3 = song + song2
    song3.export("output.wav", format="wav")

def init():
    if HAVE_BT:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        tts.to(device)
        
        model_id = "openai/whisper-large-v3-turbo"
        modelSR = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True
        )
        modelSR.to(device)
        processorSR = AutoProcessor.from_pretrained(model_id)
        pipe = pipeline(
        "automatic-speech-recognition",
        model=modelSR,
        tokenizer=processorSR.tokenizer,
        feature_extractor=processorSR.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    else:
        tts = 1
        modelSR = 2
        pipe = 3
    tokenizerLLM = AutoTokenizer.from_pretrained(model_path)

    modelQA = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    universalQA = pipeline(
        "text-generation",
        model=modelQA,
        tokenizer=tokenizerLLM,
    )
    
    return tts, pipe, universalQA

def answer(tts, pipe, universalQA):
    try:
        ser.write("4".encode('ascii'))
        p = pygame.mixer.Sound('listen.wav')
        p.play()
        while pygame.mixer.get_busy():
            time.sleep(0.1)
        audio_path = record_audio(device_name="sysdefault")
        
        result = pipe(audio_path, generate_kwargs={"language": "russian"})
        question = result['text']
        torch.cuda.empty_cache()
        print(question)
        tts.tts_to_file(text=f"Вы спросили {question}",
                file_path="output.wav",
                speaker_wav=file_path,
            language="ru")
        torch.cuda.empty_cache()
        
        p = pygame.mixer.Sound('output.wav')
        p.play()
        while pygame.mixer.get_busy():
            time.sleep(0.1)
        p = pygame.mixer.Sound('UWU.wav')
        p.play()
        while pygame.mixer.get_busy():
            time.sleep(0.1)
            
        audio_path = record_audio(duration=3)
        result = pipe(audio_path, generate_kwargs={"language": "russian"})
        user_response = result['text']
        print(user_response)
            
        if contains_negative_words(user_response.replace(".", "")):  # If sentiment is negative or neutral
            return
        ser.write("2".encode('ascii'))
        messages.append({"role": "user", "content": question})
        answerQA = universalQA(messages, **generation_args)
        messages.remove({"role": "user", "content": question})
    
        answer = answerQA[0]["generated_text"]
        torch.cuda.empty_cache()

        print("Ответ:", answer)
        if not HAVE_BT:
            return
        tts.tts_to_file(text=answer,
                file_path="output.wav",
                speaker_wav=file_path,
            language="ru")
        ser.write("5".encode('ascii'))
        
        p = pygame.mixer.Sound('output.wav')
        p.play()
        while pygame.mixer.get_busy():
            time.sleep(0.1)
        p = pygame.mixer.Sound('UWU.wav')
        p.play()
        print("end")
    except Exception as e:
        print(f'Ошибка {e}')

def main():  
    global last_phrase
    tts, pipe, universalQA = init()
    lasttime = time.perf_counter()
    lastface = time.perf_counter()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    with mp_face_detection.FaceDetection(min_detection_confidence=0.8) as face_detection:
        while True:
            _, frame = cap.read()
            if not _:
                break
            
            frame = cv2.flip(frame, 0)
            x, y, c = frame.shape

            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get hand landmark prediction
            result = hands.process(framergb)
            className = ''

            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)
                        landmarks.append([lmx, lmy])
                    # Drawing landmarks 
                    mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                    prediction = gesture_model.predict([landmarks])
                    classID = np.argmax(prediction)
                    className = classNames[classID]
                
                    # If thumbs up 
                    if className == 'thumbs up' and time.perf_counter()-lasttime>2 and not pygame.mixer.get_busy():
                        answer(tts, pipe, universalQA)
                        lasttime = time.perf_counter()

            # detect face
            face_results = face_detection.process(framergb)
            if face_results.detections: 
                wmax = 0
                h1max = 0
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x1, y1, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    if w > wmax:
                        wmax = w
                        h1max = x1
                    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                    cv2.putText(frame, 'Face', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                med = wmax/2+h1max
                if med > 400:
                    ser.write("3".encode('ascii'))
                    print(3, med)
                elif med < 300:
                    ser.write("1".encode('ascii'))
                    print(1, med)
                else:
                    ser.write("2".encode('ascii'))
                    print(2, med)
                lastface = time.perf_counter()
            else: 
                if time.perf_counter()-lastface > 2:
                    ser.write("6".encode('ascii'))
                    print(6)

                if(round(time.perf_counter(), 1)-round(time.perf_counter()) == 0):
                    if(random.randint(0, 100) in range(58, 60)):
                        if not pygame.mixer.get_busy():
                            verdikt = random.randint(1, 4)
                            if verdikt == last_phrase:
                                verdikt = (verdikt+1)%4+1
                            last_phrase=verdikt
                            p = pygame.mixer.Sound(f'promote{verdikt}.wav')
                            p.play()
                            while pygame.mixer.get_busy():
                                time.sleep(0.1)
                            p = pygame.mixer.Sound('UWU.wav')
                            p.play()
                            while pygame.mixer.get_busy():
                                time.sleep(0.1)

            # Show prediction 
            cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Gesture and Face Recognition", frame)

            if cv2.waitKey(1) == ord('q'):
                break

    # Release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
