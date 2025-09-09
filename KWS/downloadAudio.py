import torchaudio
from torch.utils.data import random_split
from pathlib import Path

<<<<<<< HEAD
data_root = (Path(__file__).resolve().parent.parent / "data")
=======
data_root = (Path(__file__).resolve().parent.parent / "../data/SpeechCommands/speech_commands_v0.02/data")
>>>>>>> b08b400b (	new file:   KWS/__init__.py)
data_root.mkdir(parents=True, exist_ok=True)

dataset = torchaudio.datasets.SPEECHCOMMANDS(
    root=str(data_root),
    download=True,
    subset=None
)

def filter_stop(dataset):
    stop_samples = []
    other_samples = []
    for i in range(len(dataset)):
        waveform, sample_rate, label, speaker_id, utterance_number = dataset[i]
        if label == "stop":
            stop_samples.append((waveform, 1))  # 1 = метка для "stop"
        else:
            other_samples.append((waveform, 0))  # 0 = не "stop"
    return stop_samples, other_samples

stop_data, other_data = filter_stop(dataset)
print(f"Найдено 'stop': {len(stop_data)}, других слов: {len(other_data)}")