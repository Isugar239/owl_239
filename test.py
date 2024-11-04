import telebot #type: ignore
import numpy as np
import pandas as pd #type: ignore
from sklearn.model_selection import train_test_split #type: ignore
from keras.models import Sequential, load_model#type: ignore
from keras.layers import Dense, Embedding, LSTM #type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer #type: ignore
from keras.preprocessing.sequence import pad_sequences           #type: ignore\
import os



TF_ENABLE_ONEDNN_OPTS = 0
bot = telebot.TeleBot('7935579912:AAHV6o-3WeKVJIqixeMy0VPEljtZE3l7J3Y')
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
y = df['label'].map(labels_mapping).values

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=10, input_length=X.shape[1]))
model.add(LSTM(10))
model.add(Dense(5, activation='softmax'))

# Проверка на наличие файла 'model1.keras'
if os.path.exists('model1.keras'):
    model.load_weights('model1.keras')  # Загрузка весов модели из файла
    print("Модель успешно загружена из файла model1.keras")
else:
    # Обучение модели
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1)
    model.save_weights('model1.keras')  # Сохранение весов модели в файл
    print("Модель обучена и сохранена в файл model1.keras")

# Оценка точности модели
accuracy = model.evaluate(X_test, y_test)
print(f'Точность модели: {accuracy[1] * 100:.2f}%')

# if os.path.exists('model1.keras'):
#     model = model.load('model1.keras')  
#     model.add(Dense(5, activation='softmax'))
#     model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# else:
#     model = Sequential()
#     model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=10, input_length=X.shape[1]))
#     model.add(LSTM(10))
#     model.add(Dense(5, activation='softmax'))
#     model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=1)
#     model.save('model1.keras')  # Сохранение весов модели
# accuracy = model.evaluate(X_test, y_test)
# print(f'Точность модели: {accuracy[1] * 100:.2f}%')



@bot.message_handler(content_types=['text'])

def test(message):
    def classify_query(query):
        seq = tokenizer.texts_to_sequences([query])
        padded = pad_sequences(seq, maxlen=X.shape[1])
        pred = model.predict(padded)
        return list(labels_mapping.keys())[np.argmax(pred)]
    if(message.text!='/start' and message.chat.id != 1612279056):
        new_query = message.text
        with open('users', 'r+') as file:
                users = file.readlines()
                if f'{message.from_user.username}|{message.chat.id}\n' not in users:
                    file.write(f'{message.from_user.username}|{message.chat.id}\n')   
        result = classify_query(new_query)
        print(f'Запрос: "{new_query}" относится к: {result}')
        bot.send_message(message.chat.id, f'Запрос: "{new_query}" относится к: {result}')
        bot.send_message(1612279056, new_query)
        bot.send_message(1612279056, f'от @{message.from_user.username}: относится к: {result}')
    else:
        if(message.chat.id == 1612279056 and message.reply_to_message != 'None'):
            verdikt = message.text 
            result = classify_query(message.reply_to_message.text)
            if verdikt == '1':
                with open('requests', 'a', encoding='utf-8') as f:
                    f.write(f'\n{message.reply_to_message.text}')
                with open('topic', 'a', encoding='utf-8') as f:
                    f.write(f'\n{result}')
            else:
                match verdikt.split():
                    case['сотрудник']:
                        with open('topic', 'a', encoding='utf-8') as f:
                            f.write(f'\n{verdikt}')
                    case['история места']:
                        with open('topic', 'a', encoding='utf-8') as f:
                            f.write(f'\n{verdikt}')
                    case['местоположение']:
                        with open('topic', 'a', encoding='utf-8') as f:
                            f.write(f'\n{verdikt}')
                    case['проект']:
                        with open('topic', 'a', encoding='utf-8') as f:
                            f.write(f'\n{verdikt}')   
                    case['инфа']:
                        with open('topic', 'a', encoding='utf-8') as f:
                            f.write(f'\n{verdikt}')     
                    case _:
                        verdikt = '0' 
                if verdikt != '0':
                    with open('requests', 'a', encoding='utf-8') as f:
                        f.write(f'\n{message.reply_to_message.text}')
bot.polling(non_stop=True, interval=0)