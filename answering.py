from transformers import *

data = {

}
with open('requests', 'r+', encoding='utf-8') as f:
    data["text"] = (f.read().split("\n"))
with open('topic', 'r+', encoding='utf-8') as f:
    data["label"] = (f.read().split("\n"))
print(data)

df = pd.DataFrame(data)
verdikt = ''
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
X = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(X)

labels_mapping = {'местоположение': 0, 'история места': 1, 'сотрудник': 2, 'проект' : 3, 'инфа': 4}
topics = [

]
for item in os.listdir("files"):
    if os.path.isdir(os.path.join("dukes", item)):
        topics.append(item)
y = df['label'].map(labels_mapping).values

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


if os.path.exists('model1.keras') and input('relearn or use last version') != 'relearn':  
    model = load_model('model1.keras')  
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
else:
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=10, input_length=X.shape[1]))
    model.add(LSTM(10))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1)
    model.save('model1.keras')  # Сохранение весов модели
accuracy = model.evaluate(X_test, y_test)

model = pipeline(model="seara/rubert-tiny2-russian-sentiment")
def emotion(text):
    result = model(text)
    return emoji, result[0]['label']
def answer(text):
    pass
