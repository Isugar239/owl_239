import torch
import pickle
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer, util
import time

model_id = "microsoft/Phi-4-mini-instruct"
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(embedding_model)
model = AutoModel.from_pretrained(embedding_model)

tokenizer = AutoTokenizer.from_pretrained(model_id)
print(torch.cuda.current_device())
universalQA = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-10)
    return sum_embeddings / sum_mask

def get_embedding(text: str) -> np.ndarray:
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    return embedding[0].numpy()

def prepare_chunks(text_path: str, save_to: str):
    with open(text_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(full_text)

    embeddings = [get_embedding(chunk) for chunk in chunks]

    with open(save_to, "wb") as f:
        pickle.dump((chunks, embeddings), f)

def retrieve_chunks(question: str, chunks, embeddings, top_k=3):
    q_emb = get_embedding(question)
    sims = cosine_similarity([q_emb], embeddings)[0]
    top_idx = sims.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_idx]


def generate_answer(question: str, chunks: list, embeddings: list):
    context = "\n\n".join(retrieve_chunks(question, chunks, embeddings))
    prompt = f"Если не знаешь — скажи честно. Используй только этот контекст:\n{context}\n\nВопрос: {question}\nОтвет:"
    
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
    print("Ответ:", answer)
    print(time.perf_counter()-timer)

    return answerQA[0]["generated_text"].replace(prompt, "").strip()

if __name__ == "__main__":
    import os
    embedding_path = "embeddings.pkl"
    if not os.path.exists(embedding_path):
        prepare_chunks("data.txt", embedding_path)

    with open(embedding_path, "rb") as f:
        chunks, embeddings = pickle.load(f)

    while True:
        question = input().strip()
    
        answer = generate_answer(question, chunks, embeddings)
        print("\nОтвет:", answer)
