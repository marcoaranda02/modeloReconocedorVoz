import torchaudio
import numpy as np
import shutil
from pathlib import Path

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
CARPETA_ORIGEN  = "..\dataset_limpio"        # tus audios actuales de 5 seg
CARPETA_DESTINO = "..\dataset_corto_limpio"  # nueva carpeta con segmentos de 2 seg
SAMPLE_RATE     = 16000
DURACION_SEG    = 2.0              # segundos por segmento
SOLAPAMIENTO    = 0.5              # segundos de overlap entre segmentos
DURACION_MIN    = 1.5              # descartar segmentos más cortos que esto
NORMALIZAR      = True             # normalizar volumen de cada segmento

# ─────────────────────────────────────────
# PARTIR UN AUDIO EN SEGMENTOS
# ─────────────────────────────────────────
def partir_audio(ruta, duracion_seg, solapamiento, duracion_min):
    waveform, sr = torchaudio.load(str(ruta))

    # Convertir a mono y resamplear si hace falta
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    total_samples  = waveform.shape[1]
    seg_samples    = int(duracion_seg   * SAMPLE_RATE)
    hop_samples    = int((duracion_seg - solapamiento) * SAMPLE_RATE)
    min_samples    = int(duracion_min   * SAMPLE_RATE)

    segmentos = []
    inicio    = 0

    while inicio + min_samples <= total_samples:
        fin       = min(inicio + seg_samples, total_samples)
        segmento  = waveform[:, inicio:fin]

        # Padding si el último segmento es más corto que duracion_seg
        if segmento.shape[1] < seg_samples:
            pad      = seg_samples - segmento.shape[1]
            segmento = np.pad(
                segmento.numpy(),
                ((0, 0), (0, pad)),
                mode="constant"
            )
            import torch
            segmento = torch.tensor(segmento)

        # Normalizar volumen
        if NORMALIZAR:
            import torch
            pico = segmento.abs().max()
            if pico > 0:
                segmento = segmento / pico * 0.95

        segmentos.append(segmento)
        inicio += hop_samples

    return segmentos

# ─────────────────────────────────────────
# PROCESAR TODO EL DATASET
# ─────────────────────────────────────────
def partir_dataset():
    origen  = Path(CARPETA_ORIGEN)
    destino = Path(CARPETA_DESTINO)

    # ── Limpiar destino antes de empezar ──────────────────────
    if destino.exists():
        shutil.rmtree(destino)
        print(f"  🗑️  Carpeta '{destino}' limpiada\n")
    # ──────────────────────────────────────────────────────────

    hablantes = sorted([c for c in origen.iterdir() if c.is_dir()])

    print(f"\n{'='*55}")
    print(f"  PARTIR DATASET EN SEGMENTOS DE {DURACION_SEG}s")
    print(f"  Overlap         : {SOLAPAMIENTO}s")
    print(f"  Duración mínima : {DURACION_MIN}s")
    print(f"{'='*55}\n")

    resumen = {}

    for carpeta_h in hablantes:
        nombre_h   = carpeta_h.name
        destino_h  = destino / nombre_h
        destino_h.mkdir(parents=True, exist_ok=True)

        audios      = sorted(list(carpeta_h.glob("*.wav")))
        total_segs  = 0

        print(f"  🎙️  {nombre_h}")

        for audio_path in audios:
            segmentos = partir_audio(
                audio_path,
                DURACION_SEG,
                SOLAPAMIENTO,
                DURACION_MIN
            )

            for i, seg in enumerate(segmentos):
                nombre_salida = f"{audio_path.stem}_seg{i:02d}.wav"
                ruta_salida   = destino_h / nombre_salida

                import torch
                torchaudio.save(
                    str(ruta_salida),
                    seg if isinstance(seg, torch.Tensor) else torch.tensor(seg),
                    SAMPLE_RATE
                )
                total_segs += 1

            print(f"    {audio_path.name:<30} → {len(segmentos)} segmentos")

        resumen[nombre_h] = {
            "originales" : len(audios),
            "segmentos"  : total_segs
        }

    # Resumen final
    print(f"\n{'='*55}")
    print(f"  RESUMEN")
    print(f"{'='*55}")
    print(f"  {'Hablante':<20} {'Originales':>12} {'Segmentos':>12}")
    print(f"  {'─'*45}")
    for h, datos in resumen.items():
        print(f"  {h:<20} {datos['originales']:>12} {datos['segmentos']:>12}")
    print(f"{'='*55}")
    print(f"\n✅ Dataset guardado en '{CARPETA_DESTINO}/'")
    print(f"   Apunta ECAPA-train.py a esa carpeta:")
    print(f"   CARPETA_DATASET = \"{CARPETA_DESTINO}\"")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    partir_dataset()