from transformers import pipeline, AutoTokenizer
import time
from sentence_transformers import SentenceTransformer, util
import torch

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

while True:
    try:
        question = input()

        prompt = f"Если не знаешь - открыто скажи это. основываясь ТОЛЬКО на этих данных: :\n{context}\n\n дай только ответ на этот вопрос: {question}\nОтвет:"
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

    except Exception as e:
        print(f'Ошибка: {e}')


