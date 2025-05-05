import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModel, AutoModelForCausalLM
import time
from sentence_transformers import SentenceTransformer
import sounddevice as sd
import numpy as np
import wave
import scipy
from TTS.api import TTS
import pygame
import requests
import os
import socket
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import tensorflow as tf

# Initialize pygame for audio
pygame.init()

# Configure TensorFlow to use GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Initialize MediaPipe Face Detection and Hands
mp_face_detection = mp.solutions.face_detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the hand gesture recognition model
gesture_model = load_model('mp_hand_gesture')

# Load class names for hand gestures
classNames = ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']

# Initialize voice assistant components
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

context = ''' В центре Санкт-Петербург между двумя улицами – Фурштадтской и Кирочной
    располагается Физико-математический лицей № 239. Он является одним из элитных учебных
    заведений нашего города и имеет богатую историю, как историю собственно 239 школы г.
    Ленинграда, так и историю того здания, в котором он размещается – историю одного из
    знаменитейших не только в дореволюционном Санкт-Петербурге, но и во всей России училища
    Святой Анны (Анненшуле).
    Возраст лицея, а в прошлом школы № 239, более 80 лет. Она была организована в 1918 г.
    и первоначально располагалась в "доме со львами" на углу Адмиралтейского проспекта и
    Исаакиевской площади. Школа несколько раз меняла свой адрес и в 1975 г. вселилась в здание
    по адресу Кирочная, 8а, которое ранее было построено для училища Св. Анны. Училище это в
    1986 г. отмечало свое 250-летие.  '''

messages = [
    {"role": "system", "content": f"Ты ИИ ассистент для ответов на вопросы по слеудющему тексту \n{context}\n. Если не знаешь - не пытайся угадать, признайся что не знаешь. Все цифры заменяй словами в нужном падеже. В конце ответа не ставь точку. Если в вопросе есть слово похожее на лицей, считай что это оно. основываясь ТОЛЬКО на данных, переданных в запросе дай только ответ ТОЛЬКО на этот вопрос"},
    {"role": "user", "content": "Кто директор музей 239? В конце ответа не ставь точку"},
    {"role": "assistant", "content": "Татьяна Витальевна Любченко"},
]

model_path = "microsoft/Phi-4-mini-instruct"
generation_args = {
    "max_new_tokens": 512,
    "return_full_text": False,
    "do_sample": False,
}
file_path="speaker.wav"

def initialize_voice_assistant():
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
    
    modelQA = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizerLLM = AutoTokenizer.from_pretrained(model_path)
    universalQA = pipeline(
        "text-generation",
        model=modelQA,
        tokenizer=tokenizerLLM,
    )
    
    return tts, pipe, universalQA

def process_voice_command(tts, pipe, universalQA):
    try:
        audio_path = record_audio()
        result = pipe(audio_path, generate_kwargs={"language": "russian"})
        question = result['text']
        print("Распознанный вопрос:", question)

        messages.append({"role": "user", "content": question})
        answerQA = universalQA(messages, **generation_args)
        answer = answerQA[0]["generated_text"]
        print("Ответ:", answer)
        
        tts.tts_to_file(text=answer,
                file_path="output.wav",
                speaker_wav=file_path,
            language="ru")
        p = pygame.mixer.Sound('output.wav')
        p.play()
        
    except Exception as e:
        print(f'Ошибка обработки голосовой команды: {e}')

def main():
    # Initialize voice assistant
    tts, pipe, universalQA = initialize_voice_assistant()
    
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)
    
    # Initialize face detection
    with mp_face_detection.FaceDetection(min_detection_confidence=0.7) as face_detection:
        while True:
            # Read each frame from the webcam
            success, frame = cap.read()
            if not success:
                break

            # Flip the frame vertically
            frame = cv2.flip(frame, 1)
            x, y, c = frame.shape

            # Convert the frame to RGB
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get hand landmark prediction
            result = hands.process(framergb)
            className = ''

            # Post-process the hand landmarks result
            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)
                        landmarks.append([lmx, lmy])
                    # Drawing landmarks on frames
                    mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
                    prediction = gesture_model.predict([landmarks])
                    classID = np.argmax(prediction)
                    className = classNames[classID]
                    
                    # If thumbs up gesture detected, start voice command processing
                    if className == 'thumbs up':
                        process_voice_command(tts, pipe, universalQA)

            # Perform face detection
            face_results = face_detection.process(framergb)
            if face_results.detections:
                for detection in face_results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x1, y1, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                    cv2.putText(frame, 'Face', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show the prediction on the frame
            cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("Gesture and Face Recognition", frame)

            if cv2.waitKey(1) == ord('q'):
                break

    # Release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()й
