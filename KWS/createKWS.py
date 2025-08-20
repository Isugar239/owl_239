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
        audio, sr = librosa.load(self.files[idx], sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T  # (кадры, коэффициенты)
        return torch.FloatTensor(mfcc), torch.tensor(self.labels[idx], dtype=torch.float32)

# 2. Модель
class KWSNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.pool = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.fc1 = torch.nn.Linear(64 * 11 * 2, 64)  # Подстрой под размер MFCC!
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)  # Добавляем канал
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
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
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    torch.save(model.state_dict(), "kws_stop.pth")

if __name__ == "__main__":
    train_model()