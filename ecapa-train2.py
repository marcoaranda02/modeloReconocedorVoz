import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import random
from pathlib import Path
from speechbrain.pretrained import EncoderClassifier
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ─────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────
SAMPLE_RATE     = 16000
CARPETA_DATASET = "...\dataset_corto"
MODELO_GUARDADO = "modelo_ecapa_finetuned.pt"

# Hiperparámetros de Fine-Tuning
BATCH_SIZE = 16
EPOCHS     = 15
MARGIN     = 0.4 
LR         = 1e-4
TARGET_SEC = 4.0 # Los audios se repetirán hasta alcanzar esta duración

# ─────────────────────────────────────────
# 1. EL TRUCO DEL BUCLE Y DATASET
# ─────────────────────────────────────────
def cargar_audio_con_loop(ruta, target_seconds=TARGET_SEC):
    """Carga un audio y lo repite sobre sí mismo hasta alcanzar la duración deseada."""
    wav, sr = torchaudio.load(str(ruta))
    
    # Convertir a mono y resamplear
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
        
    target_samples = int(target_seconds * SAMPLE_RATE)
    
    # Bucle: duplicar hasta alcanzar el tamaño objetivo
    while wav.shape[1] < target_samples:
        wav = torch.cat([wav, wav], dim=1)
        
    # Recortar exactamente a la longitud objetivo
    wav = wav[:, :target_samples]
    return wav.squeeze() # (tiempo,)

class TripletSpeakerDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.speakers = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        
        self.data = {}
        for spk in self.speakers:
            audios = list((self.root_dir / spk).glob("*.wav"))
            if len(audios) >= 2:
                self.data[spk] = audios
                
        self.valid_speakers = list(self.data.keys())
        self.total_files = sum(len(v) for v in self.data.values())

    def __len__(self): return self.total_files

    def __getitem__(self, idx):
        spk_a = random.choice(self.valid_speakers)
        anchor_path, positive_path = random.sample(self.data[spk_a], 2)
        
        spk_n = random.choice([s for s in self.valid_speakers if s != spk_a])
        negative_path = random.choice(self.data[spk_n])
        
        anchor_wav = cargar_audio_con_loop(anchor_path)
        positive_wav = cargar_audio_con_loop(positive_path)
        negative_wav = cargar_audio_con_loop(negative_path)
        
        return anchor_wav, positive_wav, negative_wav

# ─────────────────────────────────────────
# 2. MODELO CON CAPA EXTRA
# ─────────────────────────────────────────
class FineTuneECAPA(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": str(device)}
        )
        # Congelar base
        for param in self.ecapa.parameters():
            param.requires_grad = False
            
        # Nuestra capa para separar locutores
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

def cosine_distance(x, y):
    return 1.0 - F.cosine_similarity(x, y)

# ─────────────────────────────────────────
# 3. EXTRACCIÓN Y EVALUACIÓN (Tu lógica original)
# ─────────────────────────────────────────
def extraer_y_verificar(dataset, model, device):
    embeddings_cache = {}
    print("\n📦 Extrayendo embeddings con modelo Fine-Tuned...")
    
    model.eval()
    with torch.no_grad():
        for hablante, audios in dataset.data.items():
            embeddings_cache[hablante] = []
            for ruta in audios:
                # Usamos la misma función de loop para evaluar
                wav = cargar_audio_con_loop(ruta).unsqueeze(0).to(device)
                
                with torch.amp.autocast('cuda'):
                    emb = model(wav).squeeze().cpu()
                embeddings_cache[hablante].append(emb)
                
            print(f"  ✅ {hablante:<20} {len(audios)} embeddings")

    print("\n✅ Extracción completada!")
    
    # --- MÉTRICAS ---
    print(f"\n{'='*55}")
    print(f"  DISTANCIA INTRA-HABLANTE → debe ser BAJA  (< 0.3)")
    print(f"{'='*55}")
    todas_intra = []
    for nombre, embs in embeddings_cache.items():
        dists = []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                d = cosine_distance(embs[i].unsqueeze(0), embs[j].unsqueeze(0)).item()
                dists.append(d)
                todas_intra.append(d)
        estado = "✅" if np.mean(dists) < 0.3 else "⚠️"
        print(f"  {estado} {nombre:<15} promedio: {np.mean(dists):.4f}  max: {np.max(dists):.4f}")

    print(f"\n{'='*55}")
    print(f"  DISTANCIA INTER-HABLANTE → debe ser ALTA  (> 0.5)")
    print(f"{'='*55}")
    nombres = list(embeddings_cache.keys())
    todas_inter = []
    for i in range(len(nombres)):
        for j in range(i + 1, len(nombres)):
            embs_a = embeddings_cache[nombres[i]]
            embs_b = embeddings_cache[nombres[j]]
            dists = []
            for ea in embs_a:
                for eb in embs_b:
                    d = cosine_distance(ea.unsqueeze(0), eb.unsqueeze(0)).item()
                    dists.append(d)
                    todas_inter.append(d)
            estado = "✅" if np.mean(dists) > 0.5 else "⚠️"
            print(f"  {estado} {nombres[i]:<10} vs {nombres[j]:<10} promedio: {np.mean(dists):.4f}  min: {np.min(dists):.4f}")

    umbral = (max(todas_intra) + min(todas_inter)) / 2
    print(f"\n{'='*55}")
    print(f"  UMBRAL RECOMENDADO TRAS ENTRENAMIENTO")
    print(f"  max intra  : {max(todas_intra):.4f}")
    print(f"  min inter  : {min(todas_inter):.4f}")
    print(f"  UMBRAL     = {umbral:.4f}")
    print(f"{'='*55}")
    
    return embeddings_cache, umbral

# ─────────────────────────────────────────
# 4. BUCLE PRINCIPAL
# ─────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Usando: {device}")

    # 1. Preparar Dataset
    dataset = TripletSpeakerDataset(CARPETA_DATASET)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 2. Preparar Modelo
    print("\nCargando modelo base ECAPA-TDNN...")
    model = FineTuneECAPA(device).to(device)
    optimizer = torch.optim.Adam(model.projection.parameters(), lr=LR)
    criterion = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance, margin=MARGIN)
    scaler = torch.amp.GradScaler('cuda')
    
    # 3. ENTRENAMIENTO (Fine-Tuning)
    print(f"\n🚀 Iniciando Fine-Tuning de {EPOCHS} Epochs...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for anchor, positive, negative in pbar:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                emb_a = model(anchor)
                emb_p = model(positive)
                emb_n = model(negative)
                loss = criterion(emb_a, emb_p, emb_n)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    # 4. EVALUACIÓN Y GUARDADO
    embeddings_cache, umbral = extraer_y_verificar(dataset, model, device)
    
    print("\n📊 Construyendo referencias (centroides)...")
    referencias = {}
    for hablante, embs in embeddings_cache.items():
        centroide = torch.stack(embs).mean(dim=0)
        referencias[hablante] = F.normalize(centroide.unsqueeze(0), p=2, dim=1).squeeze()
        print(f"  ✅ {hablante}")

    torch.save({
        "referencias"  : referencias,
        "hablantes"    : list(dataset.valid_speakers),
        "umbral_rechazo": umbral,
        "modelo"       : "ecapa-finetuned",
        "projection_state_dict": model.projection.state_dict() # Vital para inferencia
    }, MODELO_GUARDADO)

    print(f"\n💾 Todo guardado en '{MODELO_GUARDADO}'")