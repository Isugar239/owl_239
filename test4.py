
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")

model = AutoModelForQuestionAnswering.from_pretrained("timpal0l/mdeberta-v3-base-squad2")
text = 'Программист проекта - Захаров Иван, ученик 8-3 класса Президентского Физико-математического лицея №239.'
question = input()
print(text)
tokenized = tokenizer.encode_plus(
   question, text,
   add_special_tokens=False
)

# Общая длина каждого блока
max_chunk_length = 512
# Длина наложения
overlapped_length = 30
# Длина вопроса в токенах
answer_tokens_length = tokenized.token_type_ids.count(0)
# Токены вопроса, закодированные числами
answer_input_ids = tokenized.input_ids[:answer_tokens_length]

# Длина основного текста первого блока без наложения
first_context_chunk_length = max_chunk_length - answer_tokens_length
# Длина основного текста остальных блоков с наложением
context_chunk_length = max_chunk_length - answer_tokens_length - overlapped_length
# Токены основного текста
context_input_ids = tokenized.input_ids[answer_tokens_length:]
# Основной текст первого блока
first = context_input_ids[:first_context_chunk_length]
# Основной текст остальных блоков
others = context_input_ids[first_context_chunk_length:]

# Если есть блоки кроме первого
# тогда обрабатываются все блоки
if len(others) > 0:
  # Кол-во нулевых токенов, для выравнивания последнего блока по длине
  padding_length = context_chunk_length - (len(others) % context_chunk_length)
  others += [0] * padding_length

  # Кол-во блоков и их длина без добавления наложения
  new_size = (
      len(others) // context_chunk_length,
      context_chunk_length
  )

  # Упаковка блоков
  new_context_input_ids = np.reshape(others, new_size)

  # Вычисление наложения
  overlappeds = new_context_input_ids[:, -overlapped_length:]
  # Добавление в наложения частей из первого блока
  overlappeds = np.insert(overlappeds, 0, first[-overlapped_length:], axis=0)
  # Удаление наложение из последнего блока, так как оно не нужно
  overlappeds = overlappeds[:-1]

  # Добавление наложения
  new_context_input_ids = np.c_[overlappeds, new_context_input_ids]
  # Добавление первого блока
  new_context_input_ids = np.insert(new_context_input_ids, 0, first, axis=0)

  # Добавление вопроса в каждый блок
  new_input_ids = np.c_[
    [answer_input_ids] * new_context_input_ids.shape[0],
    new_context_input_ids
  ]
# иначе обрабатывается только первый
else:
  # Кол-во нулевых токенов, для выравнивания блока по длине
  padding_length = first_context_chunk_length - (len(first) % first_context_chunk_length)
  # Добавление нулевых токенов
  new_input_ids = np.array(
    [answer_input_ids + first + [0] * padding_length]
  )
  # Кол-во блоков
count_chunks = new_input_ids.shape[0]

# Маска, разделяющая вопрос и текст
new_token_type_ids = [
  # вопрос блока
  [0] * answer_tokens_length
  # текст блока
  + [1] * (max_chunk_length - answer_tokens_length)
] * count_chunks

# Маска "внимания" модели на все токены, кроме нулевых в последнем блоке
new_attention_mask = (
  # во всех блоках, кроме последнего, "внимание" на все слова
  [[1] * max_chunk_length] * (count_chunks - 1)
  # в последнем блоке "внимание" только на ненулевые токены
  + [([1] * (max_chunk_length - padding_length)) + ([0] * padding_length)]
)
 # Токенизированный текст в виде блоков, упакованный в torch
new_tokenized = {
 'input_ids': torch.tensor(new_input_ids),
 'token_type_ids': torch.tensor(new_token_type_ids),
 'attention_mask': torch.tensor(new_attention_mask)
}
tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
outputs = model(**new_tokenized)
start_index = torch.argmax(outputs.start_logits)
end_index = torch.argmax(outputs.end_logits)

# Пересчёт позиций начала и конца ответа для 1D списка токенов
# = длина первого блока + (
#   позиция - длина первого блока
#   - длина ответов и отступов во всех блоках, кроме первого
# )
start_index = max_chunk_length + (
  start_index - max_chunk_length
  - (answer_tokens_length + overlapped_length)
  * (start_index // max_chunk_length)
)
end_index = max_chunk_length + (
  end_index - max_chunk_length
  - (answer_tokens_length + overlapped_length)
  * (end_index // max_chunk_length)
)

# Составление ответа
# если есть символ начала слова '▁', то он заменяется на пробел
answer = ''.join(
  [t.replace('▁', ' ') for t in tokens[start_index:end_index+1]]
)

print('Вопрос:', question)
print('Ответ:', answer)
