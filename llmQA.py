from transformers import pipeline, AutoTokenizer, AutoModelForSpeechSeq2Seq, AutoProcessor
import time
from sentence_transformers import SentenceTransformer, util
import torch
import sounddevice as sd
import numpy as np
import wave
import vlc
from TTS.api import TTS
import pygame
pygame.init()
file_path="speaker.wav"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
tts.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

def record_audio(filename="output.wav", duration=10, samplerate=46200):
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


# audio_path = record_audio()

# result = pipe(audio_path)
# print(result["text"])
timer =  time.perf_counter()

model_id = "microsoft/Phi-4-mini-instruct"
embedding_model = SentenceTransformer("ai-forever/ruElectra-small")

tokenizer = AutoTokenizer.from_pretrained(model_id)
print(torch.cuda.current_device())
universalQA = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# chunks = splitter.split_text(context)
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
1986 г. отмечало свое 250-летие. Размещение школы Св. Анны в районе Фурштадтской и
Кирочной не случайно. Училище было предназначено для детей немцев, поэтому и возникло на
территории немецкой слободы.
 Обобщающих трудов по истории Анненшуле, а тем более 239 школы и лицея нет. Есть
отдельные упоминания в журнальных статьях и сборниках. Из них наибольший интерес
представляет статья А.Б. Берлина «Питомцы знаменитой школы», опубликованная в журнале
«Ленинградская панорама».
1 Она рассказывает о наиболее выдающихся воспитанниках как
Анненшуле, так и основанной на ее базе 11-й школы. В статье также содержатся и краткие
исторические сведения по истории Анненшуле. Важным был труд В.И. Дедюлина и Г.Р.
Кипарского, отражающий основные вехи в развитии училища Святой Анны, а также сведения
по учителям и учащимся училища.
2 Интерес представляют и две статьи – В. Дедюлина и С.
Шульца, помещенные в энциклопедии: «Немцы в России». В них содержится материал по
патронам церкви Святой Анны, выделяются основные вехи развития училища.
3

 Важным источником при написании работы стали и рефераты учащихся лицея – Егора
Филатова (11-4), Екатерины Васильевой и Ольги Сергеевой (11-1), 2001 года выпуска,
посвященные истории Анненшуле и школе 239 в период блокады.
 При написании брошюры был привлечен комплекс разнохарактерных как
опубликованных, так и архивных источников. К архивным источникам, прежде всего, следует
отнести материалы музея Физико-математического лицея № 239, собранные, бережно хранимые
и любезно предоставленные директором музея Татьяной Витальевной Любченко. '''
while True:
    try:
        question = input()

        prompt = f"Если не знаешь - открыто скажи это. Все цифры заменяй словами. основываясь ТОЛЬКО на этих данных: :\n{context}\n\n дай только ответ на этот вопрос: {question}\nОтвет:"
        timer =  time.perf_counter()
        answerQA = universalQA(
            prompt,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )

        answer = answerQA[0]["generated_text"].replace(prompt, "")
        tts.tts_to_file(text=answer,
                file_path="output.wav",
                speaker_wav=file_path,
            language="ru")
        p = pygame.mixer.Sound('output.wav')
        #re.send()
        p.play()
        print("Ответ:", answer)
        print(time.perf_counter()-timer)

    except Exception as e:
        print(f'Ошибка: {e}')


