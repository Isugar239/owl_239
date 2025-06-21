#!/bin/bash
sudo mount /dev/nvme0n1p2 ~/.cache/huggingface/hub
# Устанавливаем обработчик прерывания. При выходе из скрипта (Ctrl+C) он убьет все дочерние процессы.
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

echo "Запуск TTS воркера в фоновом режиме..."
# Запускаем воркер в его собственном venv. '&' отправляет процесс в фон.
(source tts_venv/bin/activate && python3 tts_worker.py &)

# Даем воркеру пару секунд на инициализацию модели
sleep 5 

echo "Запуск основного приложения..."
# Запускаем основной скрипт. Он будет работать в текущем (основном) окружении.
python3 llmQA.py
