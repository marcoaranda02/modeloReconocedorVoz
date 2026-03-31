import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import torchaudio.transforms as T
from pathlib import Path
import random

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
# Ajusta estas rutas a tu estructura actual
DATASETS = {
    "real": Path("../../../dataset_corto"),
    "whatsapp": Path("../../../dataset_liveness_whatsapp"),
    "youtube": Path("../../../dataset_liveness_youtube")
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 20
LR = 0.0001

# ─────────────────────────────────────────
# DATASET Y PREPROCESAMIENTO
# ─────────────────────────────────────────
class LivenessDataset(Dataset):
    def __init__(self, dataset_dict):
        self.samples = []
        self.labels = {"real": 0, "whatsapp": 1, "youtube": 2}
        
        for label_name, path in dataset_dict.items():
            files = list(path.rglob("*.wav"))
            for f in files:
                self.samples.append((f, self.labels[label_name]))
        
        random.shuffle(self.samples)
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512
        )
        self.db_transform = T.AmplitudeToDB()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform, sr = torchaudio.load(path)
        
        # Generar Espectrograma de Mel
        mel_spec = self.mel_spectrogram(waveform)
        mel_db = self.db_transform(mel_spec) # [1, 64, T]
        
        # --- AJUSTE DE TAMAÑO FIJO ---
        target_h = 63  # Tamaño fijo para que el stack no falle
        res = mel_db.squeeze(0) # [64, T]
        
        if res.shape[1] < target_h:
            # Si es corto, rellenamos con ceros (padding)
            pad_size = target_h - res.shape[1]
            res = torch.nn.functional.pad(res, (0, pad_size))
        else:
            # Si es largo, recortamos
            res = res[:, :target_h]
            
        return res.transpose(0, 1), label # Retorna [63, 64]
# ─────────────────────────────────────────
# ARQUITECTURA: TRANSFORMER + MLP
# ─────────────────────────────────────────
class LivenessClassifier(nn.Module):
    def __init__(self, input_dim=64, num_classes=3):
        super(LivenessClassifier, self).__init__()
        
        # Capa de Embedding Inicial
        self.input_proj = nn.Linear(input_dim, 128)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Clasificador MLP
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: [batch, time, features]
        x = self.input_proj(x)
        x = self.transformer(x)
        
        # Global Average Pooling (promediamos el tiempo)
        x = x.mean(dim=1)
        
        return self.mlp(x)

# ─────────────────────────────────────────
# BUCLE DE ENTRENAMIENTO
# ─────────────────────────────────────────
def train():
    print(f"🚀 Entrenando en: {DEVICE}")
    dataset = LivenessDataset(DATASETS)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = LivenessClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        
        for specs, labels in train_loader:
            specs, labels = specs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            
        acc = 100 * correct / len(dataset)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss/len(train_loader):.4f} - Acc: {acc:.2f}%")

    # Guardar el modelo
    torch.save(model.state_dict(), "modelo_liveness_transformer.pt")
    print("\n✅ Entrenamiento completado. Modelo guardado como 'modelo_liveness_transformer.pt'")

if __name__ == "__main__":
    train()