import os
from pydub import AudioSegment
from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
CARPETA_WAVS = Path("..\..\..\dataset_corto")
CARPETA_MP3  = Path("..\..\..\dataset_mp3_attack")
BITRATE      = "192k"  # Calidad estándar para análisis de audio

def convertir_dataset_a_mp3():
    if not CARPETA_WAVS.exists():
        print(f"❌ No se encontró la carpeta: {CARPETA_WAVS}")
        return

    # Crear carpeta destino
    CARPETA_MP3.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Iniciando conversión a MP3 (Bitrate: {BITRATE})...")

    # Recorrer subcarpetas de hablantes (tus 10 personas)
    for subcarpeta in CARPETA_WAVS.iterdir():
        if subcarpeta.is_dir():
            destino_h = CARPETA_MP3 / subcarpeta.name
            destino_h.mkdir(parents=True, exist_ok=True)

            print(f"📦 Procesando: {subcarpeta.name}")

            for archivo_wav in subcarpeta.glob("*.wav"):
                try:
                    # Cargar WAV
                    audio = AudioSegment.from_wav(str(archivo_wav))
                    
                    # Definir nombre de salida
                    nombre_mp3 = archivo_wav.stem + ".mp3"
                    ruta_salida = destino_h / nombre_mp3

                    # Exportar a MP3
                    audio.export(str(ruta_salida), format="mp3", bitrate=BITRATE)
                    
                except Exception as e:
                    print(f"⚠️ Error en {archivo_wav.name}: {e}")

    print(f"\n✅ Conversión terminada. Archivos guardados en: {CARPETA_MP3}")

if __name__ == "__main__":
    convertir_dataset_a_mp3()