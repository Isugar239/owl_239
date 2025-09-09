import sounddevice as sd
import numpy as np
import wave
import os
import time

def record_audio(duration=5, fs=44100, channels=1):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='int16')
    time.sleep(5)
    print("Recording complete")
    return recording

def save_audio(recording, filename="output.wav", fs=44100):
    with wave.open(filename, "wb") as wf:
        num_channels = 1 if recording.ndim == 1 else recording.shape[1]
        wf.setnchannels(num_channels)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        frames = recording.tobytes()
        wf.writeframes(frames)

save_audio(record_audio())