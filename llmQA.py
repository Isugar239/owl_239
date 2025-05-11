import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCausalLM
import time
import sounddevice as sd
import numpy as np
import wave
from TTS.api import TTS
import pygame
import serial
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model # type: ignore
import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#init apc220
port = '/dev/ttyUSB0'
baud_rate = 9600
timeout = 1
ser = serial.Serial(port, baud_rate, timeout=timeout)


pygame.init()
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

mp_face_detection = mp.solutions.face_detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

gesture_model = load_model('mp_hand_gesture')

classNames = ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']

torch.random.manual_seed(0)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def record_audio(filename="voice.wav", duration=5, samplerate=46200):
    print("rec start")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    print("rec end")
    
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())
    return filename

context = ''' Президентский Физико математический лицей 239 распологается в центре санкт петербурга между фруштатской и кирочной улиц. Школа была основана в 1918 году, на данный момент ей 107 лет.
Директор лицей - Максим Яковлевич Пратусеевич. Там преподают такие предметы как: Алгебра, геометрия, иностранный язык, информатика, история, литература, русский язык, физика, Химия, биология, география, физкультура, искусство, обществознание, ОБЖ.
В лицее есть классы с 5 по 11, по несколько паралелей в каждом. Есть химбио классы и еще по многим направлениям. В конце года есть переводные экзамены. Средней бал ЕГЭ у выпускников по всем предметам около 60 баллов. В 239 много кружков на самые разные направления, включая матцентр, физцентр, робоцентр и т п, суммарно около 100. Поэтому в среднем 30-40 человек выпускаются с возможностью поступления БВИ  '''

messages = [
    {"role": "system", "content": f"Ты ИИ ассистент для ответов на вопросы по слеудющему тексту \n{context}\n. Если не знаешь - не пытайся угадать, признайся что не знаешь. Все цифры заменяй словами в нужном падеже. В конце ответа не ставь точку. Если в вопросе есть слово похожее на лицей, считай что это оно. основываясь ТОЛЬКО на данных, переданных в запросе дай только ответ ТОЛЬКО на этот вопрос"},
    {"role": "user", "content": "Кто директор 239? В конце ответа не ставь точку"},
    {"role": "assistant", "content": "Максим Яковлевич Пратусеевич"},       
    {"role": "user", "content": "Сколько человек в 11 классе?"},
    {"role": "assistant", "content": "Данной информации у меня нету"},       
]

model_path = "microsoft/Phi-4-mini-instruct"
generation_args = {
    "max_new_tokens": 512,
    "return_full_text": False,
    "do_sample": False,
}
file_path="speaker.wav"

def init():
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    tts.to(device)
    
    model_id = "openai/whisper-large-v3-turbo"
    modelSR = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True
    )
    modelSR.to(device)
    processorSR = AutoProcessor.from_pretrained(model_id)
    tokenizerLLM = AutoTokenizer.from_pretrained(model_path)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=modelSR,
        tokenizer=processorSR.tokenizer,
        feature_extractor=processorSR.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    
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
        ser.write("2".encode('ascii'))
        audio_path = record_audio()
        result = pipe(audio_path, generate_kwargs={"language": "russian"})
        question = result['text']
        print(question)

        messages.append({"role": "user", "content": question})
        answerQA = universalQA(messages, **generation_args)
        messages.remove({"role": "user", "content": question})
        import pprint
        pprint.pprint(messages)
        answer = answerQA[0]["generated_text"]
        print("Ответ:", answer)
        ser.write("3".encode('ascii'))
        tts.tts_to_file(text=answer,
                file_path="output.wav",
                speaker_wav=file_path,
            language="ru")
        p = pygame.mixer.Sound('output.wav')
        p.play()
    except Exception as e:
        print(f'Ошибка {e}')

def main():
    tts, pipe, universalQA = init()
    lasttime = time.perf_counter()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    with mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection:
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
                ser.write("0".encode('ascii'))
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x1, y1, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                    cv2.putText(frame, 'Face', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                ser.write("5".encode('ascii'))
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
