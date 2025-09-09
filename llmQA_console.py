import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from gtts import gTTS
import pygame
import time

EMBEDDING_MODEL_NAME = "ai-forever/sbert_large_nlu_ru"
model_path = "microsoft/Phi-4-mini-instruct"
generation_args = {
    "max_new_tokens": 182,
    "return_full_text": False,
    "do_sample": False,
}

# Эмбеддинги
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Загрузка базы знаний
KNOWLEDGE_VECTOR_DATABASE = FAISS.load_local(
    folder_path="./znania",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

tokenizerLLM = AutoTokenizer.from_pretrained(model_path)
modelQA = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
)
universalQA = pipeline(
    "text-generation",
    model=modelQA,
    tokenizer=tokenizerLLM,
)

pygame.init()

while True:
    question = input()
    if question.strip().lower() == 'exit':
        break
    similar_chunks = KNOWLEDGE_VECTOR_DATABASE.similarity_search_with_score(question, k=3)
    context = ""
    for i, (chunk, score) in enumerate(similar_chunks, 1):
        context += chunk.page_content
    messages = [
            {"role": "system", "content": f"Ты сова, находиштся на РРО - росийской робототехнической олимпиаде в категории Искуственный интеллект. Ты отвечаешь на вопросы по тексту \n{context}\n. Если не знаешь - не пытайся угадать, признайся что не знаешь. Пытайся отвечать коротко, но не упуская детали"},
            {"role": "user", "content": "Кто директор 239? В конце ответа не ставь точку"},
            {"role": "assistant", "content": "Максим Яковлевич Пратусевич"},       
            {"role": "user", "content": "Сколько человек в 11 классе?"},
            {"role": "assistant", "content": "Данной информации у меня нет"},       
        ]
    messages.append({"role": "user", "content": question})
    answerQA = universalQA(messages, **generation_args)
    answer = answerQA[0]["generated_text"]
    print("Ответ:", answer)
    # Озвучка через gTTS
    tts = gTTS(answer, lang="ru")
    tts.save("tts_console.mp3")
    p = pygame.mixer.Sound("tts_console.mp3")
    p.play()
    while pygame.mixer.get_busy():
        time.sleep(0.1) 