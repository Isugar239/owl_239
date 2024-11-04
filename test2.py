from transformers import pipeline

# Загрузка модели для понимания смысла текста
model_name = 'DeepPavlov/rubert-base-cased'
nlp = pipeline('text-classification', model=model_name)
text = "Ваш текст здесь"

def nastroy(text):
    result = nlp(text)
    result = result[0]
    if result['label']=='LABEL_0':
        emoji = 'Negative'
    elif result['label']=='LABEL_1':
        emoji = 'Neutral'
    else:
        emoji = 'Positive'
    return emoji, result['score']
flag = True
while flag:
    try:
        print(nastroy(input()))
    except KeyboardInterrupt:
        flag = False

