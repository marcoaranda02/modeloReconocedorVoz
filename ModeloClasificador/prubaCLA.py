import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import sounddevice as sd
import numpy as np

# ─────────────────────────────────────────
# CONFIGURACIÓN TÉCNICA
# ─────────────────────────────────────────
MODEL_PATH = "modelo_liveness_transformer.pt"
SAMPLE_RATE_MIC = 48000    # Tu Yeti Nano
SAMPLE_RATE_MODEL = 16000  # Frecuencia del modelo
DURACION_GRAB = 2.0        # Segundos de captura
DEVICE_ID = 2              # ID de tu micro

CLASES = {0: "REAL (EN VIVO) ✅", 1: "ATAQUE: WHATSAPP 📱", 2: "ATAQUE: YOUTUBE 📺"}

# ─────────────────────────────────────────
# ARQUITECTURA (Debe ser idéntica a la entrenada)
# ─────────────────────────────────────────
class LivenessClassifier(nn.Module):
    def __init__(self, input_dim=64, num_classes=3):
        super(LivenessClassifier, self).__init__()
        self.input_proj = nn.Linear(input_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=8, dim_feedforward=256, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.mlp(x.mean(dim=1))

# ─────────────────────────────────────────
# PROCESAMIENTO DE AUDIO
# ─────────────────────────────────────────
def preparar_espectrograma(audio_np):
    # 1. Resample y Normalización
    wav = torch.from_numpy(audio_np.squeeze()).unsqueeze(0)
    resampler = T.Resample(orig_freq=SAMPLE_RATE_MIC, new_freq=SAMPLE_RATE_MODEL)
    wav_16k = resampler(wav)
    
    pico = wav_16k.abs().max()
    if pico > 0: wav_16k = wav_16k / pico * 0.95

    # 2. Generar Mel Spectrogram
    mel_trans = T.MelSpectrogram(sample_rate=16000, n_mels=64, n_fft=1024, hop_length=512)
    db_trans = T.AmplitudeToDB()
    
    mel_db = db_trans(mel_trans(wav_16k)).squeeze(0).transpose(0, 1) # [T, 64]
    
    # 3. Ajuste de tamaño fijo (Padding/Crop)
    target_h = 63
    if mel_db.shape[0] < target_h:
        pad_size = target_h - mel_db.shape[0]
        mel_db = torch.nn.functional.pad(mel_db, (0, 0, 0, pad_size))
    else:
        mel_db = mel_db[:target_h, :]
        
    return mel_db.unsqueeze(0) # [1, 63, 64]

# ─────────────────────────────────────────
# BUCLE DE PRUEBA
# ─────────────────────────────────────────
def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LivenessClassifier().to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"✅ Modelo cargado exitosamente en {device}.")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return

    print("\n--- TESTER DE LIVENESS (TRANSFORMER) ---")
    while True:
        input("\n🎤 Presiona ENTER para probar una muestra (2 seg)...")
        print("🔴 Escuchando...")
        
        grabacion = sd.rec(int(DURACION_GRAB * SAMPLE_RATE_MIC), 
                           samplerate=SAMPLE_RATE_MIC, channels=1, dtype='float32', device=DEVICE_ID)
        sd.wait()
        
        with torch.no_grad():
            spec = preparar_espectrograma(grabacion).to(device)
            output = model(spec)
            probs = torch.nn.functional.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        print(f"\n🎯 RESULTADO: {CLASES[pred]}")
        print(f"📊 Confianza: {probs[0][pred]*100:.2f}%")
        print("-" * 40)

if __name__ == "__main__":
    test()