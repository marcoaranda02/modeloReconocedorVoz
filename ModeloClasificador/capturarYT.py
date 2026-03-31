import sounddevice as sd
import numpy as np
import torchaudio
import torch
import os
from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURACIÓN DINÁMICA
# ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
FOLDER_YT = BASE_DIR / "dataset_liveness_youtube"

SAMPLE_RATE_MIC = 48000    # Yeti Nano
SAMPLE_RATE_MODEL = 16000  # Frecuencia objetivo
DURACION_SEG = 2.0         # Tamaño de cada muestra final
DEVICE_ID = 2              # ID de tu micro

def segmentar_y_guardar(audio_16k, nombre_base):
    """Parte un audio largo en trozos de 2 segundos."""
    samples_por_seg = int(DURACION_SEG * SAMPLE_RATE_MODEL)
    total_samples = audio_16k.shape[1]
    
    num_segmentos = total_samples // samples_por_seg
    
    for i in range(num_segmentos):
        inicio = i * samples_por_seg
        fin = inicio + samples_por_seg
        segmento = audio_16k[:, inicio:fin]
        
        # Nombre: Video1_seg01.wav, etc.
        nombre_archivo = f"{nombre_base}_seg{i:02d}.wav"
        ruta_salida = FOLDER_YT / nombre_archivo
        torchaudio.save(str(ruta_salida), segmento, SAMPLE_RATE_MODEL)
    
    return num_segmentos

def capturar_bloque_youtube():
    FOLDER_YT.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*50)
    print("📺 CAPTURADOR MASIVO: YOUTUBE ATTACK")
    print("="*50)
    
    nombre_video = input("🔗 Identificador del video (ej. Entrevista_01): ").strip()
    segundos_grabar = float(input("⏱️ ¿Cuántos segundos quieres grabar? (ej. 60): "))
    
    input(f"\nPresiona ENTER y dale PLAY al video en YouTube...")
    print(f"🔴 Grabando {segundos_grabar} segundos...")
    
    # Grabación cruda
    grabacion = sd.rec(int(segundos_grabar * SAMPLE_RATE_MIC), 
                       samplerate=SAMPLE_RATE_MIC, channels=1, dtype='float32', device=DEVICE_ID)
    sd.wait()
    
    # Procesamiento (Resample + Normalización 0.95)
    wav = torch.from_numpy(grabacion.squeeze()).unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_freq=SAMPLE_RATE_MIC, new_freq=SAMPLE_RATE_MODEL)
    wav_16k = resampler(wav)
    
    pico = wav_16k.abs().max()
    if pico > 0:
        wav_16k = wav_16k / pico * 0.95
    
    # Segmentación automática
    total = segmentar_y_guardar(wav_16k, nombre_video)
    print(f"✅ ¡Éxito! Se generaron {total} segmentos de 2.0s.")

if __name__ == "__main__":
    try:
        while True:
            capturar_bloque_youtube()
            if input("\n¿Capturar otro video? (s/n): ").lower() != 's': break
    except KeyboardInterrupt:
        print("\n🛑 Proceso finalizado.")