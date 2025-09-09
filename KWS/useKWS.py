import torch
import torchaudio
import sounddevice as sd
import threading
import argparse
import time

class KWSNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.pool = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.fc1 = torch.nn.Linear(64 * 23 * 8, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class KWSPipeline:
    def __init__(self, model_path):
        self.sample_rate = 16000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=40,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
        )
        self.model = KWSNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _preprocess_waveform(self, waveform, sr):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        num_samples = self.sample_rate
        if waveform.size(1) < num_samples:
            pad_amount = num_samples - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        else:
            waveform = waveform[:, :num_samples]
        mfcc = self.mfcc_transform(waveform)
        mfcc = mfcc.squeeze(0).transpose(0, 1)
        return mfcc

    def __call__(self, audio_path=None, live=False, threshold=0.66):
        if live:
            self.live_prediction(threshold=threshold)
        else:
            waveform, sr = torchaudio.load(audio_path)
            mfcc = self._preprocess_waveform(waveform, sr)
            input_tensor = mfcc.unsqueeze(0).to(self.device)
            with torch.inference_mode():
                logits = self.model(input_tensor)
                prob = torch.sigmoid(logits).item()
            print("STOP" if prob > threshold else "OTHER", f"(p={prob:.3f})")

    def live_prediction(self, window_s: float = 1.0, hop_s: float = 0.25, sr: int = 16000, threshold: float = 0.66):
        print(f"Listening... window={window_s:.2f}s, hop={hop_s:.2f}s, threshold={threshold}")
        window_samples = int(window_s * sr)
        hop_samples = int(hop_s * sr)
        buffer = torch.zeros(0, dtype=torch.float32)
        try:
            while True:
                audio = sd.rec(hop_samples, samplerate=sr, channels=1, dtype='float32')
                sd.wait()
                chunk = torch.from_numpy(audio.flatten()).float()
                buffer = torch.cat([buffer, chunk])
                if buffer.numel() > window_samples:
                    buffer = buffer[-window_samples:]
                if buffer.numel() >= window_samples:
                    segment = buffer[-window_samples:]
                    mfcc = self._preprocess_waveform(segment, sr)
                    input_tensor = mfcc.unsqueeze(0).to(self.device)
                    with torch.inference_mode():
                        logits = self.model(input_tensor)
                        prob = torch.sigmoid(logits).item()
                    print("STOP" if prob > threshold else "...", f"(p={prob:.3f})")
        except KeyboardInterrupt:
            print("Stopped.")

    def start_listener(self, threshold: float = 0.66, window_s: float = 1.0, hop_s: float = 0.25, sr: int = 16000):
        stop_event = threading.Event()
        detected_event = threading.Event()
        thread = threading.Thread(
            target=self._listener_loop,
            args=(stop_event, detected_event, threshold, window_s, hop_s, sr),
            daemon=True,
        )
        thread.start()
        return stop_event, detected_event, thread

    def _listener_loop(self, stop_event: threading.Event, detected_event: threading.Event, threshold: float, window_s: float, hop_s: float, sr: int):
        window_samples = int(window_s * sr)
        hop_samples = int(hop_s * sr)
        buffer = torch.zeros(0, dtype=torch.float32)
        while not stop_event.is_set():
            audio = sd.rec(hop_samples, samplerate=sr, channels=1, dtype='float32')
            sd.wait()
            chunk = torch.from_numpy(audio.flatten()).float()
            buffer = torch.cat([buffer, chunk])
            if buffer.numel() > window_samples:
                buffer = buffer[-window_samples:]
            if buffer.numel() >= window_samples:
                segment = buffer[-window_samples:]
                mfcc = self._preprocess_waveform(segment, sr)
                input_tensor = mfcc.unsqueeze(0).to(self.device)
                with torch.inference_mode():
                    logits = self.model(input_tensor)
                    prob = torch.sigmoid(logits).item()
                if prob > threshold:
                    detected_event.set()
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kws_stop.pth', help='Путь к файлу модели .pth')
    parser.add_argument('--list-devices', action='store_true', help='Вывести список аудио-устройств и выйти')
    parser.add_argument('--device-index', type=int, default=None, help='Индекс входного устройства')
    parser.add_argument('--device-name', type=str, default=None, help='Имя входного устройства (как в query_devices)')
    parser.add_argument('--samplerate', type=int, default=16000, help='Частота дискретизации ввода')
    parser.add_argument('--threshold', type=float, default=0.1, help='Порог срабатывания')
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        raise SystemExit(0)

    sd.default.samplerate = args.samplerate
    sd.default.channels = 1

    if args.device_index is not None:
        sd.default.device = (args.device_index, None)
    elif args.device_name is not None:
        # Найти по имени
        devices = sd.query_devices()
        match = None
        for idx, dev in enumerate(devices):
            if args.device_name.lower() in str(dev['name']).lower():
                match = idx
                break
        if match is None:
            print('Устройство не найдено. Доступные устройства:')
            print(devices)
            raise SystemExit(1)
        sd.default.device = (match, None)

    pipeline = KWSPipeline(args.model)
    pipeline(live=True, threshold=args.threshold)