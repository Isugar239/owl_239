import os
import time
from TTS.api import TTS
import torch

REQUEST_FILE = "request.txt"
RESPONSE_FILE = "output.wav"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def main():
    """
    Основной цикл воркера.
    """
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)
    print("ready")
    while True:
        try:
            if os.path.exists(REQUEST_FILE) and os.path.getsize(REQUEST_FILE) > 0:
                print("new request")
                time.sleep(1)

                with open(REQUEST_FILE, 'r', encoding='utf-8') as f:
                    text_to_synthesize = f.read().strip()

                if text_to_synthesize:
                    print(f"TTS Worker: synthesize: '{text_to_synthesize}'")
                    
                    tts.tts_to_file(
                        text=text_to_synthesize,
                        file_path=RESPONSE_FILE,
                        speaker_wav="speaker.wav", 
                        language="ru"
                    )
                    
                    print("saved")
                os.remove(REQUEST_FILE)

        except Exception as e:
            print(f"error: {e}")
            if os.path.exists(REQUEST_FILE):
                os.remove(REQUEST_FILE)
        
        time.sleep(0.5)

if __name__ == "__main__":
    main() 