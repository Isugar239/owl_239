from transformers import pipeline

# Загрузка модели
qa_pipeline = pipeline("question-answering")

# Чтение файла
with open('Broshyura_Uvazhaemaya_shkola_25_avgusta.docx', 'r', encoding='utf-8') as file:
    context = file.read()

# Задаем вопрос
question = "Какова основная тема текста?"

# Получаем ответ
result = qa_pipeline(question=question, context=context)
print(result['answer'])
