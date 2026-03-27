import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import sounddevice as sd
import numpy as np
from speechbrain.pretrained import EncoderClassifier
import warnings

warnings.filterwarnings("ignore") # Para ignorar advertencias de torchaudio

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
ARCHIVO_MODELO = "modelo_ecapa_finetuned.pt"
SAMPLE_RATE    = 16000
DURACION_GRAB  = 2.0  # Tiempo que tendrá la persona para decir "presente"
TARGET_SEC     = 4.0  # El bucle interno para estabilizar la red
DEVICE_ID      = 2 # Variable global para el ID del micro
SAMPLE_RATE_MIC = 48000  # La frecuencia de tu Yeti Nano
SAMPLE_RATE_MODEL = 16000 # Lo que espera ECAPA
UMBRAL_DETECTAR_SONIDO=0.002


# ─────────────────────────────────────────
# 0. SELECCIÓN DE DISPOSITIVO (Tu lógica original)
# ─────────────────────────────────────────
def seleccionar_microfono():
    global DEVICE_ID
    print("\n--- DISPOSITIVOS DE AUDIO DETECTADOS ---")
    print(sd.query_devices())
    print("-" * 40)
    
    seleccion = input("🎤 Introduce el ID del micrófono a usar (ENTER para default): ").strip()
    if seleccion.isdigit():
        DEVICE_ID = int(seleccion)
        try:
            sd.check_input_settings(device=DEVICE_ID, samplerate=SAMPLE_RATE, channels=1)
            print(f"✅ Micrófono {DEVICE_ID} configurado correctamente.")
        except Exception as e:
            print(f"⚠️ Error: El dispositivo {DEVICE_ID} no es compatible: {e}")
            DEVICE_ID = None
    else:
        DEVICE_ID = None
        print("✅ Usando dispositivo predeterminado.")
# ─────────────────────────────────────────
# 1. LA MISMA CAPA QUE ENTRENAMOS
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

# ─────────────────────────────────────────
# 2. EL TRUCO DEL BUCLE (EN MEMORIA)
# ─────────────────────────────────────────
def procesar_audio_en_vivo(audio_numpy, target_seconds=TARGET_SEC):
    wav = torch.tensor(audio_numpy).unsqueeze(0) # (1, tiempo)
    target_samples = int(target_seconds * SAMPLE_RATE)
    
    # Repetir la señal hasta alcanzar los 4 segundos
    while wav.shape[1] < target_samples:
        wav = torch.cat([wav, wav], dim=1)
        
    wav = wav[:, :target_samples]
    return wav

# ─────────────────────────────────────────
# 3. BUCLE PRINCIPAL DE IDENTIFICACIÓN
# ─────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Iniciando motor de reconocimiento en: {device}")

    # Cargar datos
    try:
        datos = torch.load(ARCHIVO_MODELO, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{ARCHIVO_MODELO}'. Asegúrate de estar en la carpeta correcta.")
        return

    referencias = datos["referencias"]
    umbral = datos["umbral_rechazo"]
    
    # Instanciar y cargar pesos del modelo
    model = FineTuneECAPA(device).to(device)
    model.projection.load_state_dict(datos["projection_state_dict"])
    model.eval() # Modo evaluación crucial para BatchNorm
    print(f"✅ Modelo cargado. Umbral dinámico: {umbral:.4f}")
    seleccionar_microfono()
    while True:
        input("\n🎤 Presiona ENTER y di 'Presente' (tienes 2 segundos)...")
        
        # 1. Grabar a 48kHz (Configuración de tu Yeti Nano)
        print("🔴 Escuchando...")
        audio = sd.rec(int(DURACION_GRAB * SAMPLE_RATE_MIC), samplerate=SAMPLE_RATE_MIC, channels=1, dtype='float32')
        sd.wait()
        
        # Validación de silencio (Usando el audio original)
        if np.abs(audio).mean() < UMBRAL_DETECTAR_SONIDO:
            print("⚠️ No se detectó sonido suficiente. Habla más fuerte.")
            continue

        # 2. Convertir a Tensor y hacer Resample a 16kHz
        wav_tensor = torch.from_numpy(audio.squeeze()).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(orig_freq=SAMPLE_RATE_MIC, new_freq=SAMPLE_RATE_MODEL)
        wav_16k = resampler(wav_tensor)

        # 3. Estabilizar y subir a la GPU
        # IMPORTANTE: Usamos wav_16k, no el 'audio' original
        wav = procesar_audio_en_vivo(wav_16k.squeeze().numpy(), target_seconds=TARGET_SEC)
        wav = wav.to(device)

        # 4. Extraer embedding
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                emb_prueba = model(wav).squeeze()

        # ... (El resto del código de comparación de distancias se mantiene igual)

        mejor_distancia = float('inf')
        locutor_identificado = "Desconocido"
        
        print("\n📊 Distancias:")
        for locutor, ref_emb in referencias.items():
            
            # Aseguramos que la referencia también esté en la GPU
            ref_emb = ref_emb.to(device)
            
            # Distancia Coseno en GPU
            distancia = 1.0 - F.cosine_similarity(emb_prueba.unsqueeze(0), ref_emb.unsqueeze(0)).item()
            estado = "🟢" if distancia < umbral else "🔴"
            
            print(f"  {estado} {locutor:<15}: {distancia:.4f}")
            
            # Guardamos al mejor candidato que supere el umbral
            if distancia < mejor_distancia:
                mejor_distancia = distancia
                if distancia < umbral:
                    locutor_identificado = locutor

        print(f"\n{'='*40}")
        if locutor_identificado != "Desconocido":
            print(f" 🎯 IDENTIFICADO : {locutor_identificado.upper()}")
        else:
            print(" ⚠️ NO RECONOCIDO")
        print(f"{'='*40}")

if __name__ == "__main__":
    main()