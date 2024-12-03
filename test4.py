import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("timpal0l/mdeberta-v3-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("timpal0l/mdeberta-v3-base-squad2")
text = 'Говоря простым языком, языковые модели — это алгоритмы, способные продолжать тексты. Если чуть усложнить, то это вероятностные алгоритмы, и к ним сразу можно задать эмпирический критерий качества: хорошая модель даёт разумные продолжения данных ей текстов.'
# with open('text','r+', encdoing='utf-8') as f:
#   text = f.read()


def QA(question):
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
    padding_length = context_chunk_length - (len(others) % context_chunk_length)
    others += [0] * padding_length


    new_size = (
        len(others) // context_chunk_length,
        context_chunk_length
    )


    new_context_input_ids = np.reshape(others, new_size)


    overlappeds = new_context_input_ids[:, -overlapped_length:]

    overlappeds = np.insert(overlappeds, 0, first[-overlapped_length:], axis=0)

    overlappeds = overlappeds[:-1]


    new_context_input_ids = np.c_[overlappeds, new_context_input_ids]

    new_context_input_ids = np.insert(new_context_input_ids, 0, first, axis=0)


    new_input_ids = np.c_[
      [answer_input_ids] * new_context_input_ids.shape[0],
      new_context_input_ids
    ]
  # иначе обрабатывается только первый
  else:

    padding_length = first_context_chunk_length - (len(first) % first_context_chunk_length)

    new_input_ids = np.array(
      [answer_input_ids + first + [0] * padding_length]
    )

  count_chunks = new_input_ids.shape[0]


  new_token_type_ids = [

    [0] * answer_tokens_length

    + [1] * (max_chunk_length - answer_tokens_length)
  ] * count_chunks

  # Маска "внимания" модели на все токены, кроме нулевых в последнем блоке
  new_attention_mask = (

    [[1] * max_chunk_length] * (count_chunks - 1)

    + [([1] * (max_chunk_length - padding_length)) + ([0] * padding_length)]
  )

  new_tokenized = {
  'input_ids': torch.tensor(new_input_ids),
  'token_type_ids': torch.tensor(new_token_type_ids),
  'attention_mask': torch.tensor(new_attention_mask)
  }
  tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
  outputs = model(**new_tokenized)
  start_index = torch.argmax(outputs.start_logits)
  end_index = torch.argmax(outputs.end_logits)

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
  return answer
while True:
  question = input()
  print('Ответ:', QA(question))
