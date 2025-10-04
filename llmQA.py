import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCausalLM
import time
import sounddevice as sd
import numpy as np
import wave
# from TTS.api import TTS
import pygame
import random
from serial import Serial
import scipy.io.wavfile as wavfile
from scipy.signal import resample
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import num2words
import string
import re
from gtts import gTTS
from pydub import AudioSegment
from TTS.api import TTS
LOCAL_TTS = False
EMBEDDING_MODEL_NAME = "ai-forever/sbert_large_nlu_ru"
last_digit = -1
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Load the saved index
KNOWLEDGE_VECTOR_DATABASE = FAISS.load_local(
    folder_path="/media/olegg/sova/owl_239/znania",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True  # Needed for security reasons
)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#init apc220
last_phrase = 0
port = '/dev/ttyUSB0'
baud_rate = 9600
timeout = 1
while True:
    try:
        ser = Serial(port, baud_rate, timeout=timeout)
        break
    except Exception as e:
        print(e)
    
pygame.init()
# Инициализируем mixer с правильными параметрами для ALSA
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
except Exception as e:
    print(f"Mixer init error: {e}")
    # Пробуем альтернативные параметры
    try:
        pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=256)
    except Exception as e2:
        print(f"Alternative mixer init also failed: {e2}")
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def contains_negative_words(text):
    negative_words = ["нет", "не", "никак", "вряд ли", "никогда", "ни за что", "неа", "отрицательно", "исключено", "не думаю", "сомневаюсь"]
    text_lower = text.lower()
    return any(neg in text_lower for neg in negative_words)


def change_sample_rate(input_path, output_path, new_rate):
    
    rate, data = wavfile.read(input_path)

    if len(data.shape) == 2:
        num_channels = data.shape[1]
        resampled_data = np.zeros((int(data.shape[0] * new_rate / rate), num_channels))
        for i in range(num_channels):
            resampled_data[:, i] = resample(data[:, i], int(data.shape[0] * new_rate / rate))
    else:
        resampled_data = resample(data, int(data.shape[0] * new_rate / rate))
    
    # Запись нового WAV файла
    wavfile.write(output_path, new_rate, resampled_data.astype(data.dtype))

mp_face_detection = mp.solutions.face_detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

gesture_model = load_model('/media/olegg/sova/owl_239/mp_hand_gesture')
# gesture_model = keras.layers.TFSMLayere("mp_hand_gesture", call_endpoint='serving_default')
classNames = ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']

torch.random.manual_seed(0)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def record_audio(filename="/media/olegg/sova/owl_239/voice.wav", duration=5, samplerate=44100, device_name=""):
    # Переключаем Bluetooth в режим гарнитуры
    os.system("/media/olegg/sova/owl_239/bt_audio_switcher.sh start_record")
    print("rec start")
    try:
        audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
        time.sleep(duration)
        print("rec end")
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())
    except Exception as e:
        print(e)
    # Возвращаем Bluetooth в режим высокого качества
    os.system("/media/olegg/sova/owl_239/bt_audio_switcher.sh stop_record")
    return filename


model_path = "microsoft/Phi-4-mini-instruct"
generation_args = {
    "max_new_tokens": 128,
    "return_full_text": False,
    "do_sample": False,
}
file_path="/media/olegg/sova/owl_239/speaker.wav"

def do_tts(tts, text, lang):
    if tts:
        tts.tts_to_file(text=text, file_path="/media/olegg/sova/owl_239/tts_output.wav", speaker_wav=file_path, language=lang)
    else:
        tts = gTTS(text, lang=lang)
        tts.save("/media/olegg/sova/owl_239/tts_output.mp3")
        sound = AudioSegment.from_mp3("/media/olegg/sova/owl_239/temp_output.mp3")
        sound.export("/media/olegg/sova/owl_239/temp_output.wav", format="wav")

        ser.write("5".encode('ascii'))
        change_sample_rate("/media/olegg/sova/owl_239/temp_output.wav", "/media/olegg/sova/owl_239/tts_output.wav", 17000)
def init():

    model_id = "openai/whisper-large-v3"
    modelSR = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=False, use_safetensors=True
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
    
    if LOCAL_TTS:
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
    else:
        tts = 0
    return pipe, universalQA, tts


def _play_sound(sound_path: str):
    p = pygame.mixer.Sound(sound_path)
    p.play()
    while pygame.mixer.get_busy():
        time.sleep(0.05)


def _play_sound_with_gesture_interrupt(sound_path: str, cap):
    p = pygame.mixer.Sound(sound_path)
    p.play()
    while pygame.mixer.get_busy():
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue
        x, y, c = frame.shape
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(framergb)
        className = ''
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)
                    landmarks.append([lmx, lmy])
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            prediction = gesture_model.predict([landmarks])
            classID = np.argmax(prediction)
            className = classNames[classID]
            # if className == 'thumbs down':
            #     pygame.mixer.stop()
            #     ser.write("2".encode('ascii'))
            #     break
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("owl GUI", frame)
        cv2.waitKey(1)
        time.sleep(0.02)


def answer(pipe, universalQA, cap):
    # try:
        
        p = pygame.mixer.Sound('/media/olegg/sova/owl_239/listen.mp3')
        p.play()
        while pygame.mixer.get_busy():
            time.sleep(0.1)
        ser.write("4".encode('ascii'))
        audio_path = record_audio(device_name="sysdefault")
        ser.write("2".encode('ascii'))
        
        result = pipe(audio_path, generate_kwargs={"language": "russian"})
        question = result['text']

        torch.cuda.empty_cache()
        print(question)
        
        do_tts(tts, f'Вы спросили {question}?', lang="ru")
       
        
        p = pygame.mixer.Sound("/media/olegg/sova/owl_239/tts_output.wav")
        p.play()
        while pygame.mixer.get_busy():
            time.sleep(0.1)

        
        p = pygame.mixer.Sound('/media/olegg/sova/owl_239/UWU.wav')
        p.play()
        while pygame.mixer.get_busy():
            time.sleep(0.1)
        ser.write("2".encode('ascii'))
        
        audio_path = record_audio(device_name="sysdefault")
        result = pipe(audio_path, generate_kwargs={"language": "russian"})
        user_response = result['text']
        print(user_response)
        
        if contains_negative_words(user_response.replace(".", "")):  # If sentiment is negative or neutral
            return
        
        ra = random.randint(1, 4)
        p = pygame.mixer.Sound(f'/media/olegg/sova/owl_239/wait{ra}.mp3')
        p.play(loops=6)
        similar_chunks = KNOWLEDGE_VECTOR_DATABASE.similarity_search_with_score(question, k=3)
        context = ""
        for i, (chunk, score) in enumerate(similar_chunks, 1):
            context += chunk.page_content
        
        messages = [
            {"role": "system", "content": f"Ты робот-сова из города санкт-петербург. Ты отвечаешь на вопросы по тексту \n{context}\n. Если не знаешь - не пытайся угадать, признайся что не знаешь."},
            {"role": "user", "content": "Кто директор 239?"},
            {"role": "assistant", "content": "Максим Яковлевич Пратусевич"},       
            {"role": "user", "content": "Кто тебя сделал?"},
            {"role": "assistant", "content": "Захаров Иван"},       
        ]
        messages.append({"role": "user", "content": f'{question}'})
        answerQA = universalQA(messages, **generation_args)
        messages.remove({"role": "user", "content": f'{question}'})

        answer = answerQA[0]["generated_text"]
        torch.cuda.empty_cache()
        print("Ответ:", answer)

        do_tts(tts, answer, lang="ru")

        ser.write("5".encode('ascii'))
        
        pygame.mixer.stop()
        print("play")
        _play_sound_with_gesture_interrupt("/media/olegg/sova/owl_239/tts_output.wav", cap)
        
        p = pygame.mixer.Sound('/media/olegg/sova/owl_239/UWU.wav')
        p.play()
        print("end")
    # except Exception as e:
    #     print(f'Ошибка {e}')



def check_face_stable(face_detected, face_start_time, stable_duration=0.2):
    current_time = time.perf_counter()
    
    if face_detected:
        if face_start_time is None:
            return False, current_time
        elif current_time - face_start_time >= stable_duration:
            return True, face_start_time
        else:
            return False, face_start_time
    else:
        return False, None


def main(): 
    global last_digit 
    pipe, universalQA = init()
    
    ser.write("6".encode('ascii'))
    input()
    
    lasttime = time.perf_counter()
    lastface = time.perf_counter()
    
    face_start_time = None
    face_stable = False
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    try:
        with mp_face_detection.FaceDetection(min_detection_confidence=0.96) as face_detection:
            while True:
                _, frame = cap.read()
                if not _:
                    print("cant read video")
                    break
                
                x, y, c = frame.shape

                framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                result = hands.process(framergb)
                className = ''

                if result.multi_hand_landmarks:
                    landmarks = []
                    for handslms in result.multi_hand_landmarks:
                        for lm in handslms.landmark:
                            lmx = int(lm.x * x)
                            lmy = int(lm.y * y)
                            landmarks.append([lmx, lmy])
                        mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                        prediction = gesture_model.predict([landmarks])
                        classID = np.argmax(prediction)
                        className = classNames[classID]
                        
                        if className == 'thumbs up' and time.perf_counter()-lasttime>2 and not pygame.mixer.get_busy():
                            ser.write("2".encode('ascii'))
                            
                            answer(pipe, universalQA, cap)
                            lasttime = time.perf_counter()

                face_results = face_detection.process(framergb)
                face_detected = face_results.detections is not None and len(face_results.detections) > 0
                
                face_stable, face_start_time = check_face_stable(face_detected, face_start_time, stable_duration=1.0)
                
                if face_stable: 
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
                        cv2.putText(frame, 'Stable', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    med = wmax/2+h1max
                    if med < 300:
                        # if last_digit==3:
                        #     continue
                        # last_digit = 3
                        ser.write("3".encode('ascii'))
                        print(3, med)
                    elif med > 450:
                        # if last_digit==1:
                        #     continue
                        # last_digit=1
                        ser.write("1".encode('ascii'))
                        print(1, med)
                    else:
                        # if last_digit==2:
                        #     continue
                        # last_digit=2
                        ser.write("2".encode('ascii'))
                        print(2, med)
                    lastface = time.perf_counter()
                elif face_detected:
                    for detection in face_results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        ih, iw, _ = frame.shape
                        x1, y1, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 165, 255), 2)
                        cv2.putText(frame, 'Detecting...', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                    ser.write("2".encode('ascii'))
                else: 
                    if time.perf_counter()-lastface > 2:
                        # if last_digit==6:
                        #     continue
                        # last_digit=6
                        ser.write("6".encode('ascii'))
                        print(6)
                    else:
                        ser.write("2".encode('ascii'))

                cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow("owl GUI", frame)

                if cv2.waitKey(1) == ord('q'):
                    break
    except Exception as e:
        print(f"Main loop error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
