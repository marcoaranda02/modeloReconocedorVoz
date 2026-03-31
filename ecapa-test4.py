import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import sounddevice as sd
import numpy as np
import librosa
from speechbrain.pretrained import EncoderClassifier
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────
ECAPA_MODEL_PATH      = "modelo_ecapa_finetuned.pt"
LIVENESS_MODEL_PATH   = "modelo_liveness_transformer.pt"
SAMPLE_RATE_MIC       = 48000
SAMPLE_RATE_MODEL     = 16000
DURACION_GRAB         = 2.0
TARGET_SEC            = 4.0
DEVICE_ID             = 2
UMBRAL_DETECTAR_SONIDO = 0.002

CLASES_LIVENESS = {0: "REAL (EN VIVO)", 1: "ATAQUE: WHATSAPP", 2: "ATAQUE: YOUTUBE"}

# ─────────────────────────────────────────
# ARQUITECTURAS
# ─────────────────────────────────────────

class LivenessClassifier(nn.Module):
    def __init__(self, input_dim=64, num_classes=3):
        super().__init__()
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


class FineTuneECAPA(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": str(device)}
        )
        self.projection = nn.Sequential(
            nn.Linear(192, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Linear(192, 192)
        )

    def forward(self, x):
        with torch.no_grad():
            emb = self.ecapa.encode_batch(x).squeeze(1)
        out = self.projection(emb)
        out = F.normalize(out, p=2, dim=1)
        return out

# ─────────────────────────────────────────
# PROCESAMIENTO DE AUDIO
# ─────────────────────────────────────────

def recortar_silencio_inicial(audio_np, umbral=0.005):
    indices = np.where(np.abs(audio_np) > umbral)[0]
    if len(indices) > 0:
        inicio = indices[0]
        margen = int(0.01 * SAMPLE_RATE_MIC)
        return audio_np[max(0, inicio - margen):]
    return audio_np

def validar_biometria_dsp(audio_np, sr):
    mfcc       = librosa.feature.mfcc(y=audio_np, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_var   = np.mean(np.var(mfcc_delta, axis=1))

    S       = np.abs(librosa.stft(audio_np))
    entropy = np.mean(librosa.feature.spectral_flatness(S=S))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_np, sr=sr, roll_percent=0.85))

    if mfcc_var < 10.0:
        return False, f"Textura artificial detectada (Var: {mfcc_var:.2f})"
    if rolloff < 5000:
        return False, f"Frecuencias altas recortadas ({int(rolloff)}Hz)"
    if entropy > 0.04:
        return False, "Firma de audio procesado detectada"

    return True, "DSP OK"

def extraer_metricas_basicas(audio_np, sr):
    rms      = np.sqrt(np.mean(audio_np**2))
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio_np, sr=sr))
    return {"RMS": rms, "Centroide": centroid}

def preparar_espectrograma(audio_np):
    """Prepara el Mel Spectrogram [1, 63, 64] para el modelo de liveness."""
    wav = torch.from_numpy(audio_np.squeeze()).unsqueeze(0)

    resampler = T.Resample(orig_freq=SAMPLE_RATE_MIC, new_freq=SAMPLE_RATE_MODEL)
    wav_16k   = resampler(wav)

    pico = wav_16k.abs().max()
    if pico > 0:
        wav_16k = wav_16k / pico * 0.95

    mel_trans = T.MelSpectrogram(sample_rate=SAMPLE_RATE_MODEL, n_mels=64, n_fft=1024, hop_length=512)
    db_trans  = T.AmplitudeToDB()
    mel_db    = db_trans(mel_trans(wav_16k)).squeeze(0).transpose(0, 1)  # [T, 64]

    target_h = 63
    if mel_db.shape[0] < target_h:
        mel_db = F.pad(mel_db, (0, 0, 0, target_h - mel_db.shape[0]))
    else:
        mel_db = mel_db[:target_h, :]

    return mel_db.unsqueeze(0)  # [1, 63, 64]

def procesar_para_ecapa(audio_numpy):
    """Prepara el tensor de 4s repetido para ECAPA."""
    wav = torch.tensor(audio_numpy).unsqueeze(0)
    target_samples = int(TARGET_SEC * SAMPLE_RATE_MODEL)
    while wav.shape[1] < target_samples:
        wav = torch.cat([wav, wav], dim=1)
    return wav[:, :target_samples]

def seleccionar_microfono():
    global DEVICE_ID
    print("\n--- DISPOSITIVOS DE AUDIO DETECTADOS ---")
    print(sd.query_devices())
    print("-" * 40)
    sel = input(f"🎤 Introduce ID (ENTER para ID {DEVICE_ID}): ").strip()
    if sel.isdigit():
        DEVICE_ID = int(sel)
    print(f"✅ Usando dispositivo ID: {DEVICE_ID}")

# ─────────────────────────────────────────
# BUCLE PRINCIPAL
# ─────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Motor iniciado en: {device}")

    # — Cargar modelo de liveness —
    liveness_model = LivenessClassifier().to(device)
    try:
        liveness_model.load_state_dict(torch.load(LIVENESS_MODEL_PATH, map_location=device))
        liveness_model.eval()
        print(f"✅ Liveness model cargado.")
    except FileNotFoundError:
        print(f"❌ No se encontró '{LIVENESS_MODEL_PATH}'. Saliendo.")
        return

    # — Cargar modelo ECAPA —
    try:
        datos = torch.load(ECAPA_MODEL_PATH, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"❌ No se encontró '{ECAPA_MODEL_PATH}'. Saliendo.")
        return

    referencias = datos["referencias"]
    umbral_id   = datos["umbral_rechazo"]
    ecapa_model = FineTuneECAPA(device).to(device)
    ecapa_model.projection.load_state_dict(datos["projection_state_dict"])
    ecapa_model.eval()
    print(f"✅ ECAPA model cargado. Hablantes: {list(referencias.keys())}")

    seleccionar_microfono()

    resampler_ecapa = torchaudio.transforms.Resample(
        orig_freq=SAMPLE_RATE_MIC, new_freq=SAMPLE_RATE_MODEL
    )

    while True:
        input("\n🎤 Presiona ENTER y di 'Presente'...")
        print("🔴 Escuchando...")

        audio_raw = sd.rec(
            int(DURACION_GRAB * SAMPLE_RATE_MIC),
            samplerate=SAMPLE_RATE_MIC, channels=1, dtype='float32', device=DEVICE_ID
        )
        sd.wait()

        # 1. RECORTAR SILENCIO INICIAL
        audio_np = recortar_silencio_inicial(audio_raw.squeeze(), umbral=0.005)

        # 2. VALIDACIONES BÁSICAS
        if len(audio_np) < int(0.5 * SAMPLE_RATE_MIC):
            print("⚠️  Audio demasiado corto después del recorte.")
            continue
        if np.abs(audio_np).mean() < UMBRAL_DETECTAR_SONIDO:
            print("⚠️  No se detectó sonido suficiente.")
            continue

        # 3. CAPA 1 — FILTROS DSP
        pasa_dsp, msg_dsp = validar_biometria_dsp(audio_np, SAMPLE_RATE_MIC)
        if not pasa_dsp:
            print(f"🛡️  [BLOQUEADO · DSP] {msg_dsp}")
            continue

        metricas = extraer_metricas_basicas(audio_np, SAMPLE_RATE_MIC)
        print(f"✅ DSP OK | RMS: {metricas['RMS']:.4f} | Centroide: {int(metricas['Centroide'])}Hz")

        # 4. CAPA 2 — LIVENESS CLASSIFIER (Transformer)
        with torch.no_grad():
            spec   = preparar_espectrograma(audio_raw).to(device)
            output = liveness_model(spec)
            probs  = F.softmax(output, dim=1)
            pred   = torch.argmax(probs, dim=1).item()
            conf   = probs[0][pred].item() * 100

        print(f"🔍 Liveness: {CLASES_LIVENESS[pred]} ({conf:.1f}%)")

        if pred != 0:
            print(f"🛡️  [BLOQUEADO · LIVENESS] {CLASES_LIVENESS[pred]}")
            continue

        # 5. RECONOCIMIENTO — ECAPA
        wav_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        wav_16k    = resampler_ecapa(wav_tensor)
        wav        = procesar_para_ecapa(wav_16k.squeeze().numpy()).to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                emb_prueba = ecapa_model(wav).squeeze()

        mejor_distancia = float('inf')
        loc_id = "Desconocido"

        for locutor, ref_emb in referencias.items():
            dist = 1.0 - F.cosine_similarity(
                emb_prueba.unsqueeze(0), ref_emb.to(device).unsqueeze(0)
            ).item()
            if dist < mejor_distancia:
                mejor_distancia = dist
                if dist < umbral_id:
                    loc_id = locutor

        print(f"\n🎯 RESULTADO: {loc_id.upper()} (Dist: {mejor_distancia:.4f})")
        print("=" * 45)


if __name__ == "__main__":
    main()