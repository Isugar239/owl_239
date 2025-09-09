import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

# 1. Подготовка данных
class SpeechCommandsDataset(Dataset):
    def __init__(self, root_dir, target_word="stop"):
        self.root_dir = root_dir
        self.files = []
        self.labels = []
<<<<<<< HEAD
=======
        self.sample_rate = 16000
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=40,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
        )
>>>>>>> b08b400b (	new file:   KWS/__init__.py)
        
        for word in os.listdir(root_dir):
            word_dir = os.path.join(root_dir, word)
            if not os.path.isdir(word_dir):
                continue
            for file in os.listdir(word_dir):
                if file.endswith(".wav"):
                    self.files.append(os.path.join(word_dir, file))
                    self.labels.append(1 if word == target_word else 0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
<<<<<<< HEAD
        audio, sr = librosa.load(self.files[idx], sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T  # (кадры, коэффициенты)
=======
        waveform, sr = torchaudio.load(self.files[idx])
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        # Паддинг/обрезка до ровно 1 сек
        num_samples = self.sample_rate
        if waveform.size(1) < num_samples:
            pad_amount = num_samples - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        else:
            waveform = waveform[:, :num_samples]
        # MFCC: [1, n_mfcc, time] -> [time, n_mfcc]
        mfcc = self.mfcc_transform(waveform)
        mfcc = mfcc.squeeze(0).transpose(0, 1)
>>>>>>> b08b400b (	new file:   KWS/__init__.py)
        return torch.FloatTensor(mfcc), torch.tensor(self.labels[idx], dtype=torch.float32)

# 2. Модель
class KWSNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.pool = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3))
<<<<<<< HEAD
        self.fc1 = torch.nn.Linear(64 * 11 * 2, 64)  # Подстрой под размер MFCC!
=======
        self.fc1 = torch.nn.Linear(64 * 23 * 8, 64)  # для входа [batch, 1, 98, 40]
>>>>>>> b08b400b (	new file:   KWS/__init__.py)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Добавляем канал
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
<<<<<<< HEAD
        return torch.sigmoid(self.fc2(x))

# 3. Обучение
def train_model():
    dataset = SpeechCommandsDataset("data/")
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = KWSNet()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
=======
        return self.fc2(x)  # логиты

# 3. Обучение
def train_model():
    dataset = SpeechCommandsDataset("/home/sova/data/SpeechCommands/speech_commands_v0.02")
    if len(dataset) == 0:
        raise ValueError(f"Датасет пуст: нет .wav в {dataset.root_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    
    model = KWSNet().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Начальный лосс на первом батче
    first_batch = next(iter(train_loader))
    with torch.no_grad():
        init_logits = model(first_batch[0].to(device, non_blocking=True))
        init_loss = criterion(init_logits, first_batch[1].to(device, non_blocking=True).unsqueeze(1)).item()
        print(f"Initial batch loss: {init_loss:.4f}")

    for epoch in range(1):
        for step, (batch_x, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(batch_x.to(device, non_blocking=True))
            loss = criterion(logits, batch_y.to(device, non_blocking=True).unsqueeze(1))
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                with torch.no_grad():
                    pred = torch.sigmoid(logits)
                    print(f"Step {step}: loss={loss.item():.4f}, pred_mean={pred.mean().item():.4f}, y_mean={batch_y.float().mean().item():.4f}")
>>>>>>> b08b400b (	new file:   KWS/__init__.py)
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    torch.save(model.state_dict(), "kws_stop.pth")

if __name__ == "__main__":
    train_model()