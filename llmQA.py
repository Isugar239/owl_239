import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoModelForCausalLM
import time
import sounddevice as sd
import numpy as np
import wave
import pygame
import random
from serial import Serial
import scipy.io.wavfile as wavfile
from scipy.signal import resample
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model  # type: ignore
import tensorflow as tf
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from gtts import gTTS
from pydub import AudioSegment
from TTS.api import TTS

# ========== CFG ==========
LOCAL_TTS = False
CLEAR_CACHE = True
CONFIRM_QUESTION = True
EMBEDDING_MODEL_NAME = "ai-forever/sbert_large_nlu_ru"

# ========== BASE DIRECTORY ==========
base_dir = os.path.dirname(os.path.abspath(__file__))

torch.random.manual_seed(0)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

KNOWLEDGE_VECTOR_DATABASE = FAISS.load_local(
    folder_path=os.path.join(base_dir, "znania"),
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
except:
    pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=256)

mp_face_detection = mp.solutions.face_detection
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
gesture_model = load_model(os.path.join(base_dir, "mp_hand_gesture"))
classNames = ['okay', 'peace', 'thumbs up', 'thumbs down', 'call me', 'stop', 'rock', 'live long', 'fist', 'smile']


def record_audio(filename=None, duration=5, samplerate=44100):
    if filename is None:
        filename = os.path.join(base_dir, "voice.wav")
    print("rec start")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    time.sleep(duration)
    print("rec end")
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())
    return filename


def change_sample_rate(input_path, output_path, new_rate):
    rate, data = wavfile.read(input_path)
    resampled_data = resample(data, int(data.shape[0] * new_rate / rate))
    wavfile.write(output_path, new_rate, resampled_data.astype(data.dtype))


def do_tts(tts, text, lang):
    output_wav = os.path.join(base_dir, "tts_output.wav")
    temp_mp3 = os.path.join(base_dir, "temp_output.mp3")
    temp_wav = os.path.join(base_dir, "temp_output.wav")

    if LOCAL_TTS:
        tts.tts_to_file(text=text, file_path=output_wav, speaker_wav="", language=lang)
    else:
        tts = gTTS(text, lang=lang)
        tts.save(temp_mp3)
        sound = AudioSegment.from_mp3(temp_mp3)
        sound.export(temp_wav, format="wav")
        change_sample_rate(temp_wav, output_wav, 17000)


def init():
    model_id = "openai/whisper-large-v3"
    modelSR = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=False)
    modelSR.to(device)
    processorSR = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline("automatic-speech-recognition",
                    model=modelSR,
                    tokenizer=processorSR.tokenizer,
                    feature_extractor=processorSR.feature_extractor,
                    torch_dtype=torch_dtype,
                    device=device)

    tokenizerLLM = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")
    modelQA = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-mini-instruct",
                                                   device_map=device,
                                                   torch_dtype=torch_dtype,
                                                   trust_remote_code=True)
    universalQA = pipeline("text-generation", model=modelQA, tokenizer=tokenizerLLM)

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True) if LOCAL_TTS else 0
    return pipe, universalQA, tts


def answer(pipe, universalQA, cap, tts):
    ser.write("1\n".encode('ascii'))
    p = pygame.mixer.Sound(os.path.join(base_dir, "listen.mp3"))
    p.play()
    while pygame.mixer.get_busy():
        time.sleep(0.1)

    audio_path = record_audio()
    ser.write("2\n".encode('ascii'))

    result = pipe(audio_path, generate_kwargs={"language": "russian"})
    question = result['text']
    print(question)

    if CONFIRM_QUESTION:
        do_tts(tts, f'Вы спросили {question}?', lang="ru")
        p = pygame.mixer.Sound(os.path.join(base_dir, "tts_output.wav"))
        p.play()
        while pygame.mixer.get_busy():
            time.sleep(0.1)

        ser.write("5\n".encode('ascii'))
        p = pygame.mixer.Sound(os.path.join(base_dir, "UWU.wav"))
        p.play()
        while pygame.mixer.get_busy():
            time.sleep(0.1)

        ser.write("1\n".encode('ascii'))
        audio_path = record_audio()
        result = pipe(audio_path, generate_kwargs={"language": "russian"})
        user_response = result['text']
        if "нет" in user_response.lower():
            return

    ser.write("2\n".encode('ascii'))
    ra = random.randint(1, 4)
    p = pygame.mixer.Sound(os.path.join(base_dir, f"wait{ra}.mp3"))
    p.play()

    similar_chunks = KNOWLEDGE_VECTOR_DATABASE.similarity_search_with_score(question, k=3)
    context = "".join([c.page_content for c, _ in similar_chunks])

    messages = [
        {"role": "system", "content": f"Ты робот-сова из Лицея 239. Используй текст:\n{context}\n"},
        {"role": "user", "content": question},
    ]

    answerQA = universalQA(messages, max_new_tokens=128, return_full_text=False)
    answer = answerQA[0]["generated_text"]

    ser.write("3\n".encode('ascii'))
    do_tts(tts, answer, lang="ru")
    p = pygame.mixer.Sound(os.path.join(base_dir, "tts_output.wav"))
    p.play()
    while pygame.mixer.get_busy():
        time.sleep(0.05)

    ser.write("4\n".encode('ascii'))
    p = pygame.mixer.Sound(os.path.join(base_dir, "UWU.wav"))
    p.play()
    while pygame.mixer.get_busy():
        time.sleep(0.05)
    print("end answer")


def control_head_with_p_regulator(face_center, frame_center, prev_value=125, Kp=0.8, dead_zone=30, smooth=0.3):
    error = face_center - frame_center

    if abs(error) < dead_zone:
        target_value = 125
    else:
        target_value = int(125 + Kp * error)
        target_value = max(5, min(255, target_value))

    smoothed = int(prev_value * (1 - smooth) + target_value * smooth)
    return smoothed


def main():
    pipe, universalQA, tts = init()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

    frame_center = 400
    head_value = 125
    lasttime = time.perf_counter()

    with mp_face_detection.FaceDetection(min_detection_confidence=0.96) as face_detection:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_detection.process(framergb)
            face_detected = face_results.detections is not None and len(face_results.detections) > 0

            if face_detected:
                best_face = max(face_results.detections, key=lambda d: d.location_data.relative_bounding_box.width)
                bbox = best_face.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x1, y1, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
                face_center = x1 + w // 2

                head_value = control_head_with_p_regulator(face_center, iw // 2, head_value)
                ser.write(f"{head_value}\n".encode('ascii'))

                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{head_value}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                ser.write("125\n".encode('ascii'))

            cv2.imshow("owl GUI", frame)
            if cv2.waitKey(1) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
