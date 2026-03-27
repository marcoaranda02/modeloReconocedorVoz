import torch
import torchaudio
import numpy as np
import sounddevice as sd
import librosa

# Configuración para tu Yeti Nano
SAMPLE_RATE_MIC = 48000 

def extraer_metricas(audio_np, sr):
    # 1. Energía RMS (Volumen real de la señal)
    rms = np.sqrt(np.mean(audio_np**2))
    
    # 2. Zero Crossing Rate (Ayuda a detectar ruido vs voz)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio_np))
    
    # 3. Centroide Espectral (Indica dónde está el "brillo" del audio)
    # Las grabaciones baratas suelen perder frecuencias altas.
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_np, sr=sr))
    
    # 4. Relación Señal-Ruido (Estimada)
    ancho_banda = np.mean(librosa.feature.spectral_bandwidth(y=audio_np, sr=sr))

    return {
        "RMS": rms,
        "ZCR": zcr,
        "Centroide": spectral_centroid,
        "Ancho_Banda": ancho_banda
    }

def analizar_en_vivo(duracion=2.0):
    print(f"🔴 Grabando {duracion}s para análisis de seguridad...")
    audio = sd.rec(int(duracion * SAMPLE_RATE_MIC), samplerate=SAMPLE_RATE_MIC, channels=1, dtype='float32')
    sd.wait()
    audio_flat = audio.flatten()
    
    metricas = extraer_metricas(audio_flat, SAMPLE_RATE_MIC)
    
    print("\n--- CARACTERÍSTICAS DETECTADAS ---")
    for k, v in metricas.items():
        print(f"📊 {k:<15}: {v:.6f}")
    
    return metricas

if __name__ == "__main__":
    # Prueba esto hablando normal y luego poniendo una grabación de tu celular frente al micro
    analizar_en_vivo()