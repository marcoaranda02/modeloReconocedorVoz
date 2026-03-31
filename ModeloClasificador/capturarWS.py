import sounddevice as sd
import numpy as np
import torchaudio
import torch
import os
from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURACIÓN DINÁMICA DE RUTAS
# ─────────────────────────────────────────
# Sube 3 niveles: de ModeloClasificador -> modelo2 -> modeloECAPA2seg -> Raíz
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
FOLDER_WHATSAPP = BASE_DIR / "dataset_liveness_whatsapp"

# Parámetros técnicos
SAMPLE_RATE_MIC = 48000    # Tu Yeti Nano
SAMPLE_RATE_MODEL = 16000  # Frecuencia para el Transformer/ECAPA
DURACION_GRAB = 2.0        # Ajustado a tu dataset_corto
DEVICE_ID = 2              # ID de tu micro

# ─────────────────────────────────────────
# UTILIDADES DE AUDIO
# ─────────────────────────────────────────

def procesar_y_limpiar(audio_np):
    """Convierte a 16kHz y normaliza al 0.95."""
    wav = torch.from_numpy(audio_np.squeeze()).unsqueeze(0)
    
    # Resample a 16kHz para coincidir con el modelo
    resampler = torchaudio.transforms.Resample(orig_freq=SAMPLE_RATE_MIC, new_freq=SAMPLE_RATE_MODEL)
    wav_16k = resampler(wav)
    
    # Normalización idéntica a tu script de segmentación
    pico = wav_16k.abs().max()
    if pico > 0:
        wav_16k = wav_16k / pico * 0.95
        
    return wav_16k

def capturar_liveness_whatsapp():
    if not FOLDER_WHATSAPP.exists():
        FOLDER_WHATSAPP.mkdir(parents=True, exist_ok=True)
        print(f"📁 Carpeta creada en: {FOLDER_WHATSAPP}")

    print("\n" + "="*50)
    print("🚀 GENERADOR DE ATAQUES: WHATSAPP (2.0s @ 16kHz)")
    print("="*50)
    
    persona = input("👤 Nombre de la persona (ej. Persona01): ").strip()
    persona_path = FOLDER_WHATSAPP / persona
    persona_path.mkdir(parents=True, exist_ok=True)

    print(f"\nPreparado para {persona}. Reproduce en WhatsApp frente al micro.")
    
    total_muestras = 15 
    
    for i in range(1, total_muestras + 1):
        input(f"\n[Muestra {i}/{total_muestras}] ENTER para grabar...")
        print("🔴 ESCUCHANDO...")
        
        grabacion = sd.rec(int(DURACION_GRAB * SAMPLE_RATE_MIC), 
                           samplerate=SAMPLE_RATE_MIC, 
                           channels=1, 
                           dtype='float32', 
                           device=DEVICE_ID)
        sd.wait()
        
        wav_final = procesar_y_limpiar(grabacion)
        
        nombre_archivo = f"{persona}_wa_attack_{i:02d}.wav"
        ruta_salida = persona_path / nombre_archivo
        torchaudio.save(str(ruta_salida), wav_final, SAMPLE_RATE_MODEL)
        
        print(f"✅ Guardado en dataset_liveness_whatsapp/{persona}/{nombre_archivo}")

if __name__ == "__main__":
    try:
        sd.check_input_settings(device=DEVICE_ID, samplerate=SAMPLE_RATE_MIC, channels=1)
        while True:
            capturar_liveness_whatsapp()
            if input("\n¿Capturar otra persona? (s/n): ").lower() != 's': break
    except Exception as e:
        print(f"❌ Error: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Proceso detenido.")