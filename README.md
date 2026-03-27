# 🎤 Speaker Recognition System (ECAPA-TDNN)

Sistema de **reconocimiento de locutor** basado en **ECAPA-TDNN**, con:

* 🔍 Extracción de embeddings (SpeechBrain)
* 🧠 Fine-tuning con Triplet Loss
* 🛡️ Detección de audio falso (anti-spoofing)
* 🎙️ Reconocimiento en tiempo real desde micrófono

---

## 📦 Requisitos

* Anaconda o Miniconda instalado
* GPU opcional (CUDA recomendado)
* Micrófono funcional

---

## ⚙️ Instalación del entorno

Este proyecto usa un entorno de Conda definido en `environment.yml`.

### 1. Clonar repositorio

```bash
git clone https://github.com/tu_usuario/tu_repo.git
cd tu_repo
```

---

### 2. Crear entorno con Anaconda

```bash
conda env create -f environment.yml
```

---

### 3. Activar entorno

```bash
conda activate speaker_env
```

---

### 4. Verificar instalación

```bash
python -c "import torch; print(torch.__version__)"
```

---

## 🚀 Ejecución del sistema

### 🔊 1. Ejecutar reconocimiento en vivo

```bash
python ecapa-test3.py
```

---

### 🎤 2. Seleccionar micrófono

El sistema mostrará los dispositivos disponibles:

```bash
Introduce ID (ENTER para ID 2):
```

---

### 🗣️ 3. Hablar

Presiona ENTER y di:

```
"Presente"
```

---

## 🧠 Flujo del sistema

1. 🎙️ Captura de audio (2 segundos)
2. ✂️ Recorte de silencio inicial
3. 🛡️ Validación biométrica:

   * MFCC + delta
   * Entropía espectral
   * Spectral rolloff
   * RMS / ZCR
4. 🔄 Resample a 16 kHz
5. 🤖 Extracción de embedding (ECAPA)
6. 📏 Comparación con centroides
7. 🎯 Clasificación final

---

## 🛡️ Seguridad (Anti-Spoofing)

El sistema bloquea:

* 🔊 Audios reproducidos (bocinas/celular)
* 🤖 Voces sintéticas
* 🔇 Señales demasiado limpias o planas

Ejemplo:

```
🛡️ [ACCESO BLOQUEADO] Textura artificial detectada
```

---

## 📁 Estructura del proyecto

```
.
├── ecapa-test3.py              # Sistema principal (tiempo real)
├── AnalisisAudio.py           # Métricas de audio
├── ecapa-train2.py            # Entrenamiento
├── evaluar_modelo.py          # Evaluación
├── modelo_ecapa_finetuned.pt  # Modelo entrenado
├── environment.yml            # Entorno Conda
└── dataset_limpio/            # Audios por hablante
```

---

## 🧪 Entrenamiento (opcional)

```bash
python ecapa-train2.py
```

---

## 📊 Evaluación

```bash
python evaluar_modelo.py
```

---

## ❗ Problemas comunes

### ❌ `CommandNotFoundError: conda`

Solución:

* Asegúrate de tener Anaconda instalado
* Reinicia la terminal

---

### ❌ Error con PyTorch / CUDA

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

---

### ❌ No detecta micrófono

* Ejecuta:

```python
import sounddevice as sd
print(sd.query_devices())
```

---

## ✨ Notas

* El modelo usa **SpeechBrain ECAPA preentrenado**
* Solo se entrena la **projection layer**
* Los embeddings son de **192 dimensiones**

---

## 📌 Autor

Marco Antonio Aranda Sainz

---

## ⭐ Recomendación

Para mejores resultados:

* Graba en ambiente silencioso
* Usa el mismo micrófono para entrenamiento e inferencia
* Evita ruido de fondo

---
