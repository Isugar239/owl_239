import telebot #type: ignore
import numpy as np
import pandas as pd #type: ignore
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 
from keras.models import Sequential, load_model #type: ignore
from keras.layers import Dense, Embedding, LSTM #type: ignore
from keras.preprocessing.sequence import pad_sequences   
from tensorflow.keras.preprocessing.text import Tokenizer  #type: ignore
import os
from answering import *
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

bot = telebot.TeleBot('7935579912:AAHV6o-3WeKVJIqixeMy0VPEljtZE3l7J3Y')


if(input('tg or local?') == 'tg'):

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
else:
    flag = True
    def classify_query(query):
                    seq = tokenizer.texts_to_sequences([query])
                    padded = pad_sequences(seq, maxlen=X.shape[1])
                    pred = model.predict(padded)
                    return list(labels_mapping.keys())[np.argmax(pred)]
    while flag:
        quest = input()
        try:          
            result = classify_query(quest)
            print(result, emotion(quest))
        except KeyboardInterrupt:
            flag = False
            
