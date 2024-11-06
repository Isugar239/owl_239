from transformers import pipeline
model = pipeline(model="seara/rubert-tiny2-russian-sentiment")
print(model("Привет, ты мне нравишься!"))

def nastroy(text):
    result = model(text)
    result = result[0]
    if result['label']=='negative':
        emoji = 'Negative'
    elif result['label']=='neutral':
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

