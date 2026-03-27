import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import sounddevice as sd
import numpy as np
import librosa
from speechbrain.pretrained import EncoderClassifier
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────
ARCHIVO_MODELO        = "modelo_ecapa_finetuned.pt"
SAMPLE_RATE_MIC       = 48000  
SAMPLE_RATE_MODEL     = 16000  
DURACION_GRAB         = 2.0    
TARGET_SEC            = 4.0    
DEVICE_ID             = 2      
UMBRAL_DETECTAR_SONIDO = 0.002

# ─────────────────────────────────────────
# CLASES Y FUNCIONES DE PROCESAMIENTO
# ─────────────────────────────────────────

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

def recortar_silencio_inicial(audio_np, umbral=0.005):
    """Elimina el ruido muerto del inicio para limpiar métricas."""
    indices = np.where(np.abs(audio_np) > umbral)[0]
    if len(indices) > 0:
        inicio = indices[0]
        margen = int(0.01 * SAMPLE_RATE_MIC) # 10ms de margen
        return audio_np[max(0, inicio - margen):]
    return audio_np
def validar_biometria_final(audio_np, sr):
    # 1. MFCC Delta Var (Subimos el umbral de 8.0 a 12.0)
    # Tus logs muestran que las grabaciones andan en 4-7. Tu voz real debería estar arriba de 12.
    mfcc = librosa.feature.mfcc(y=audio_np, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_var = np.mean(np.var(mfcc_delta, axis=1))

    # 2. Entropía Espectral (Mide la complejidad del caos humano)
    # Las grabaciones digitales son más 'ordenadas' y predecibles.
    S = np.abs(librosa.stft(audio_np))
    entropy = np.mean(librosa.feature.spectral_flatness(S=S))

    # 3. Rolloff (Mantenemos en 5000Hz para detectar el recorte de agudos)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_np, sr=sr, roll_percent=0.85))

    # --- NUEVOS UMBRALES ESTRICTOS ---
    if mfcc_var < 10.0: # Ajuste basado en tus logs (bloqueaste 7.14, vamos por más)
        return False, f"Textura artificial detectada (Var: {mfcc_var:.2f})"

    if rolloff < 5000:
        return False, f"Frecuencias altas recortadas ({int(rolloff)}Hz)"

    # Si la señal es demasiado 'limpia' o plana, es digital
    if entropy > 0.04: 
        return False, "Firma de audio procesado detectada"

    return True, "Biometría Humana Confirmada"
def extraer_metricas_basicas(audio_np, sr):
    rms = np.sqrt(np.mean(audio_np**2))
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio_np, sr=sr))
    return {"RMS": rms, "Centroide": centroid}

def procesar_audio_en_vivo(audio_numpy, target_seconds=TARGET_SEC):
    wav = torch.tensor(audio_numpy).unsqueeze(0)
    target_samples = int(target_seconds * SAMPLE_RATE_MODEL)
    while wav.shape[1] < target_samples:
        wav = torch.cat([wav, wav], dim=1)
    return wav[:, :target_samples]

def seleccionar_microfono():
    global DEVICE_ID
    print("\n--- DISPOSITIVOS DE AUDIO DETECTADOS ---")
    print(sd.query_devices())
    print("-" * 40)
    sel = input(f"🎤 Introduce ID (ENTER para ID {DEVICE_ID}): ").strip()
    if sel.isdigit(): DEVICE_ID = int(sel)
    print(f"✅ Usando dispositivo ID: {DEVICE_ID}")

# ─────────────────────────────────────────
# BUCLE PRINCIPAL
# ─────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Motor iniciado en: {device}")

    try:
        datos = torch.load(ARCHIVO_MODELO, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró '{ARCHIVO_MODELO}'.")
        return

    referencias = datos["referencias"]
    umbral_id = datos["umbral_rechazo"]
    model = FineTuneECAPA(device).to(device)
    model.projection.load_state_dict(datos["projection_state_dict"])
    model.eval()

    seleccionar_microfono()

    while True:
        input("\n🎤 Presiona ENTER y di 'Presente'...")
        print("🔴 Escuchando...")
        
        # Captura cruda
        audio_raw = sd.rec(int(DURACION_GRAB * SAMPLE_RATE_MIC), samplerate=SAMPLE_RATE_MIC, channels=1, dtype='float32')
        sd.wait()
        
        # 1. RECORTAR SILENCIO INICIAL
        audio_np = recortar_silencio_inicial(audio_raw.squeeze(), umbral=0.005)
        
        # 2. VALIDACIÓN DE INTEGRIDAD
        if len(audio_np) < int(0.5 * SAMPLE_RATE_MIC):
            print("⚠️ Audio demasiado corto después del recorte.")
            continue

        if np.abs(audio_np).mean() < UMBRAL_DETECTAR_SONIDO:
            print("⚠️ No se detectó sonido suficiente.")
            continue

        # 3. CAPA DE SEGURIDAD (MFCC + Rolloff)
        pasa_seguridad, mensaje = validar_biometria_final(audio_np, SAMPLE_RATE_MIC)
        
        if not pasa_seguridad:
            print(f"🛡️  [ACCESO BLOQUEADO] {mensaje}")
            continue
        
        metricas = extraer_metricas_basicas(audio_np, SAMPLE_RATE_MIC)
        print(f"✅ {mensaje} | RMS: {metricas['RMS']:.4f} | Centroide: {int(metricas['Centroide'])}Hz")

        # 4. RECONOCIMIENTO (ECAPA)
        wav_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=SAMPLE_RATE_MIC, new_freq=SAMPLE_RATE_MODEL)
        wav_16k = resampler(wav_tensor)
        
        wav = procesar_audio_en_vivo(wav_16k.squeeze().numpy()).to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                emb_prueba = model(wav).squeeze()

        mejor_distancia = float('inf')
        loc_id = "Desconocido"
        
        for locutor, ref_emb in referencias.items():
            dist = 1.0 - F.cosine_similarity(emb_prueba.unsqueeze(0), ref_emb.to(device).unsqueeze(0)).item()
            if dist < mejor_distancia:
                mejor_distancia = dist
                if dist < umbral_id: loc_id = locutor

        print(f"\n🎯 RESULTADO: {loc_id.upper()} (Dist: {mejor_distancia:.4f})")
        print("="*45)

if __name__ == "__main__":
    main()