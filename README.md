# VoxList — Sistema Biométrico de Asistencia por Voz

Sistema de asistencia que combina tres capas de seguridad:
detección de vivacidad (anti-replay), verificación semántica (Whisper)
y reconocimiento de hablante (ECAPA-TDNN).

---

## Índice

1. [Arquitectura del pipeline](#1-arquitectura-del-pipeline)
2. [Metodología: clasificador de liveness](#2-metodología-clasificador-de-liveness)
3. [Estructura del repositorio](#3-estructura-del-repositorio)
4. [Requisitos e instalación](#4-requisitos-e-instalación)
5. [Configuración inicial (una sola vez)](#5-configuración-inicial-una-sola-vez)
6. [Arranque del servidor](#6-arranque-del-servidor)
7. [Exportar dependencias](#7-exportar-dependencias)
8. [Solución de problemas](#8-solución-de-problemas)

---

## 1. Arquitectura del pipeline

```
Audio del alumno (2 seg desde su celular)
        │
        ▼
┌───────────────────────────────────────┐
│  Capa 0 — Preprocesamiento            │
│  · Recorte de silencio inicial        │
│  · Validación de energía RMS          │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│  Capa 1 — Filtros DSP  (librosa)      │
│  · Varianza delta-MFCC                │
│  · Spectral rolloff                   │
│  · Spectral flatness                  │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│  Capa 2 — Liveness Classifier         │
│  Transformer + MLP  (propio)          │
│  · Entrada: Mel Spectrogram [63, 64]  │
│  · Salida: real / WhatsApp / YouTube  │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│  Capa 3 — Whisper  (HuggingFace)      │
│  · openai/whisper-small               │
│  · Verifica que se dijo "presente"    │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│  Capa 4 — ECAPA-TDNN  (SpeechBrain)  │
│  · Embedding 192 dims                 │
│  · Similitud coseno vs referencias   │
│  · Fine-tuneado con 10 hablantes     │
└──────────────────┬────────────────────┘
                   │
                   ▼
          Supabase — attendance
```

Cola FIFO: los audios se procesan de uno en uno. Si varios
alumnos graban al mismo tiempo, cada uno espera su turno.
El badge "N en cola" del panel del profesor se actualiza cada 2s.

---

## 2. Metodología: clasificador de liveness

Esta sección documenta cómo se construyó el modelo
`modelo_liveness_transformer.pt` desde cero.

### 2.1 Problema

Detectar si el audio de entrada proviene de una voz humana
en vivo o de un ataque de replay (reproducción grabada).
Se definieron 3 clases:

| Clase | Etiqueta | Descripción |
|-------|----------|-------------|
| 0 | `real` | Voz capturada directamente del hablante |
| 1 | `whatsapp` | Audio reproducido desde un mensaje de WhatsApp |
| 2 | `youtube` | Audio reproducido desde un video de YouTube |

### 2.2 Recolección del dataset

**Clase 0 — real** (`dataset_corto/`)

Grabaciones directas de los 10 hablantes del sistema.
~15 segmentos de 2 segundos por persona.

**Clase 1 — WhatsApp** (`dataset_liveness_whatsapp/`)
Script: `capturarWS.py`

```
Proceso:
  1. Se envía un audio del hablante por WhatsApp
  2. Se reproduce el mensaje desde el teléfono frente al micrófono
  3. El script graba 2 segundos, remuestrea a 16 kHz y guarda

Parámetros:
  SAMPLE_RATE_MIC   = 48000 Hz  (Yeti Nano)
  SAMPLE_RATE_MODEL = 16000 Hz
  DURACION_GRAB     = 2.0 s
  Muestras/persona  = 15
  Normalización     = pico / 0.95
  Salida            = {persona}_wa_attack_{N}.wav
```

**Clase 2 — YouTube** (`dataset_liveness_youtube/`)
Script: `capturarYT.py`

```
Proceso:
  1. Se reproduce un video de YouTube con la voz del hablante
  2. El script graba N segundos continuos frente al micrófono
  3. El audio se remuestrea a 16 kHz y se segmenta en clips de 2 s

Parámetros:
  Duración configurable por consola (ej. 60 segundos)
  Segmentación automática en clips de 2.0 s exactos
  Salida: {VideoID}_seg00.wav, _seg01.wav, ...
```

**Complemento — ataques vía MP3** (`procesarWAV.py`)

Convierte el dataset real a MP3 a 192 kbps para simular
la compresión que introduce WhatsApp/Telegram al reenviar audios:

```
dataset_corto/ (WAV)  →  dataset_mp3_attack/ (MP3 192k)
```

### 2.3 Preprocesamiento de audio

Todos los audios pasan por la misma cadena antes de entrar al modelo:

```
WAV (16 kHz, mono)
        │
        ▼
MelSpectrogram
  sample_rate = 16000
  n_mels      = 64
  n_fft       = 1024
  hop_length  = 512
        │
        ▼
AmplitudeToDB  →  escala logarítmica
        │
        ▼
Reshape [1, 64, T] → transponer → [T, 64]
        │
        ▼
Padding / crop a longitud fija → [63, 64]
        │
        ▼
Input al modelo: [batch, 63, 64]
```

### 2.4 Arquitectura del modelo

```python
LivenessClassifier(
  input_proj  : Linear(64 → 128)

  transformer : TransformerEncoder(
                  d_model        = 128
                  nhead          = 8
                  dim_feedforward = 256
                  num_layers     = 3
                  batch_first    = True
                )

  pooling     : x.mean(dim=1)   # global average pooling temporal

  mlp         : Linear(128 → 64)
                ReLU()
                Dropout(0.3)
                Linear(64 → 3)  # 3 clases
)
```

Parámetros de entrenamiento:

```
Optimizer : Adam
LR        : 0.0001
Loss      : CrossEntropyLoss
Batch     : 16
Epochs    : 20
Device    : CUDA (RTX 4050)
Script    : train_liveness.py
Salida    : modelo_liveness_transformer.pt
```

### 2.5 Evaluación y prueba en tiempo real

Script: `prubaCLA.py`

```bash
python prubaCLA.py
# Presiona ENTER → habla o reproduce audio
# RESULTADO: REAL (EN VIVO) ✅  |  Confianza: 97.3%
```

### 2.6 Umbrales DSP calibrados para audio móvil

Los codecs móviles (Opus/AAC) alteran el espectro respecto al
micrófono de PC, por lo que `server.py` usa umbrales relajados:

| Métrica | Umbral PC | Umbral móvil | Razón del ajuste |
|---------|-----------|--------------|------------------|
| Varianza delta-MFCC | > 10.0 | > 3.0 | Clips cortos comprimen la dinámica vocal |
| Spectral rolloff | > 5000 Hz | > 2500 Hz | Opus/AAC limitan las frecuencias altas |
| Spectral flatness | < 0.04 | < 0.15 | Compresión eleva la planitud espectral |

Los ataques de replay siguen siendo detectables porque su
firma espectral es aún más extrema que estos umbrales relajados.

---

## 3. Estructura del repositorio

```
VersionCompletav2/
├── server.py                        Backend FastAPI (pipeline + cola FIFO)
├── index.html                       Frontend profesor + alumno (rol por URL)
├── identificador_completo.py        Pruebas locales con micrófono del PC
├── modelo_ecapa_finetuned.pt        ECAPA-TDNN fine-tuneado (10 hablantes)
├── modelo_liveness_transformer.pt   Clasificador de vivacidad entrenado
├── capturarWS.py                    Captura ataques WhatsApp para el dataset
├── capturarYT.py                    Captura ataques YouTube para el dataset
├── procesarWAV.py                   Convierte WAV a MP3 (simula compresión)
├── train_liveness.py                Entrenamiento del clasificador liveness
└── prubaCLA.py                      Prueba del clasificador en tiempo real
```

---

## 4. Requisitos e instalación

### Paquetes Python

```bash
pip install fastapi uvicorn python-multipart soundfile \
            torch torchaudio speechbrain librosa supabase \
            pydub transformers numpy
```

### Herramientas del sistema

| Herramienta | Uso | Instalación Windows |
|-------------|-----|---------------------|
| ffmpeg | Decodificar WebM/M4A de móviles | `winget install ffmpeg` |
| ngrok | Túnel público para alumnos | `winget install ngrok` |

### Archivos de modelos requeridos

Los archivos `.pt` deben estar en la raíz del proyecto.
No se incluyen en el repositorio por su tamaño.

---

## 5. Configuración inicial (una sola vez)

### 5.1 Autenticar ngrok

```bash
ngrok config add-authtoken TU_TOKEN_AQUI
```

Token disponible en: https://dashboard.ngrok.com

### 5.2 Actualizar constraint de Supabase

En el SQL Editor del proyecto de Supabase:

```sql
ALTER TABLE attendance
DROP CONSTRAINT attendance_method_check;

ALTER TABLE attendance
ADD CONSTRAINT attendance_method_check
CHECK (method = ANY (
  ARRAY['voice'::text, 'manual'::text, 'voice_mobile'::text]
));
```

### 5.3 Whisper

Se descarga automáticamente la primera vez (~500 MB).
Queda cacheado en `%USERPROFILE%\.cache\huggingface\`.

---

## 6. Arranque del servidor

### Terminal 1 — FastAPI

```bash
cd VersionCompletav2
conda activate speaker_env
uvicorn server:app --host 0.0.0.0 --port 8000
```

Esperar los 4 mensajes de confirmación:

```
✅ Liveness cargado.
✅ ECAPA cargado. Hablantes: ['Marco2', 'diego', ...]
✅ Whisper cargado.
✅ Cola de audio iniciada.
```

> Primera vez: varios minutos por la descarga de Whisper.
> Siguientes veces: ~30 segundos.

### Terminal 2 — ngrok

```bash
ngrok http 8000          # si está en el PATH
.\ngrok.exe http 8000    # si el .exe está en la carpeta
```

Copiar la URL que aparece:

```
Forwarding  https://abc123.ngrok-free.app -> http://localhost:8000
```

### En el navegador del profesor

1. Abrir `http://localhost:8000`
2. **Config** → URL del servidor FastAPI → pegar URL de ngrok → Guardar
3. Crear sesión → el QR aparece automáticamente
4. Los alumnos escanean y dicen **"Presente"** desde su celular

---

## 7. Exportar dependencias

```bash
conda activate speaker_env

# Opción A — entorno completo (recomendado para reproducir exacto)
conda env export > environment.yml
# Recrear en otra máquina:
conda env create -f environment.yml

# Opción B — solo pip (más portable entre sistemas operativos)
pip freeze > requirements.txt
# Instalar en otra máquina:
pip install -r requirements.txt

# Opción C — solo paquetes conda
conda list --export > conda_packages.txt
```

> Para compartir con usuarios en Linux o Mac, usar la Opción B.
> Los `.yml` de conda incluyen builds específicos de Windows.

---

## 8. Solución de problemas

| Problema | Causa probable | Solución |
|----------|----------------|----------|
| `Error 23514 attendance_method_check` | Constraint desactualizado | Ejecutar el ALTER TABLE del paso 5.2 |
| `"ngrok" no se reconoce` | ngrok no está en el PATH | Usar `.\ngrok.exe` o `winget install ngrok` |
| Warning `attention mask is not set` | Comportamiento normal de Whisper | Inofensivo, no afecta el resultado |
| Alumno bloqueado por DSP | Ruido de fondo o codec agresivo | Lugar silencioso, sin altavoz |
| Alumno bloqueado por Whisper | No se entendió "presente" | Hablar claro y cerca del micrófono |
| Pipeline tarda más de 10 segundos | Modelo corriendo en CPU | Verificar `Dispositivo: cuda` al arrancar |
| GPU no detectada | PyTorch sin soporte CUDA | `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| QR no abre la página correcta | URL de ngrok no configurada | Pegar URL ngrok en Config y guardar |
| QR deja de funcionar | ngrok gratuito cambia URL al reiniciar | Actualizar Config con la nueva URL |
