from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import torch
from torchaudio import transforms
import torch.nn as nn
import io
import torch.nn.functional as F
import soundfile as sf

app = FastAPI()


class UrbanAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        # Если пришел тензор размером [batch, height, width], добавляем канал
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.first(x)
        x = self.second(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.MelSpectrogram(sample_rate=22050, n_mels=64)
max_len = 500

labels = torch.load('urban_labels.pth', weights_only=True)
label_to_index = {word: ind for ind, word in enumerate(labels)}

if isinstance(labels, dict):
    index_to_label = {int(v): k for k, v in labels.items()}
elif isinstance(labels, (list, tuple)):
    index_to_label = {i: v for i, v in enumerate(labels)}
else:
    index_to_label = {i: v.item() if hasattr(v, 'item') else v for i, v in enumerate(labels)}



model = UrbanAudio()
model.load_state_dict(torch.load('urban_model.pth', map_location=device, weights_only=True))
model.to(device)
model.eval()


def change_audio(waveform, sr):
    if sr != 22050:
        resample = transforms.Resample(orig_freq=sr, new_freq=22050)
        waveform = resample(waveform)

    spec = transform(waveform)

    
    if spec.ndim == 3:
        spec = spec.squeeze(0)

    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]
    elif spec.shape[1] < max_len:
        count_len = max_len - spec.shape[1]
        spec = F.pad(spec, (0, count_len))


    return spec.unsqueeze(0)


@app.post('/predict/')
async def predict_audio(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=500, detail='Файл пустой')

    wf, sr = sf.read(io.BytesIO(data), dtype='float32')
    wf = torch.tensor(wf)

    if wf.ndim > 1:
        wf = wf.mean(dim=1)

    wf = wf.unsqueeze(0)

    spec = change_audio(wf, sr).to(device)

    with torch.no_grad():
        y_pred = model(spec)
        pred_ind = torch.argmax(y_pred, dim=1).item()

        pred_class = index_to_label.get(pred_ind, "Unknown")

    return {'Индекс': pred_ind, 'Жанр': pred_class}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
