import torch
import sounddevice as sd
import numpy as np
import wave
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

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
model1 = pipeline("question-answering", "timpal0l/mdeberta-v3-base-squad2")
model3 = pipeline("question-answering", model="AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru")
model4 = pipeline("question-answering", model="IProject-10/xlm-roberta-base-finetuned-squad2")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)
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


audio_path = record_audio()

result = pipe(audio_path)
print(result["text"])
#question = result["text"]
question = "Где располагается Физико-математический лицей № 239"
answer1 = model1(question = "отвечай подробно и только по тексту! иначе говори что не знаешь. Вопрос: " + question, context =  context)['answer']
answer3 = model3(question = "отвечай подробно и только по тексту! иначе говори что не знаешь. Вопрос: " + question, context = context)['answer']
answer4 = model4(question = "отвечай подробно и только по тексту! иначе говори что не знаешь. Вопрос: " + question, context = context)['answer']
print("mdbert ", answer1)
print("sber", answer3)
print("iProject", answer4)
