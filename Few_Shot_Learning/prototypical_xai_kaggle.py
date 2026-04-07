"""
Prototypical Networks with XAI for Few-Shot Learning - Kaggle Version
=====================================================================

This is a standalone Python script version of the Kaggle notebook.
It implements Prototypical Networks with Explainable AI (XAI) for few-shot learning.

Key fix: Using inplace=False in ReLU layers to avoid conflicts with backward hooks
used in GradCAM.

Usage:
    python prototypical_xai_kaggle.py

Requirements:
    torch>=2.0.0, torchvision>=0.15.0, numpy>=1.24.0, pandas>=1.5.0,
    scikit-learn>=1.2.0, matplotlib>=3.7.0, tqdm>=4.65.0,
    Pillow>=9.5.0, opencv-python>=4.7.0, scipy>=1.10.0
"""

import os
import sys
import random
import json
import math
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, precision_recall_fscore_support
)
from sklearn.model_selection import StratifiedShuffleSplit
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
# Use 'Agg' only if not in interactive environment
if 'DISPLAY' not in os.environ and 'KAGGLE_KERNEL_RUN_TYPE' not in os.environ:
    matplotlib.use('Agg')

warnings.filterwarnings('ignore')

# ================================================================================
# CONFIGURATION - OPTIMIZED FOR GPU UTILIZATION AND ACCURACY
# ================================================================================

@dataclass
class Config:
    """Configuration for reproducibility - OPTIMIZED for Kaggle GPU."""
    data_root: str = '/kaggle/input/few-shot-data'
    output_dir: str = '/kaggle/working'
    n_classes: int = 8
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    img_size: int = 128
    n_way: int = 8
    k_shot: int = 5
    q_query: int = 15
    q_query_eval: int = 5
    n_epochs: int = 100  # INCREASED from 30 for better convergence
    lr: float = 1e-3
    weight_decay: float = 1e-4
    embedding_dim: int = 512  # INCREASED from 128
    episodes_per_epoch: int = 100  # INCREASED from 20 for more training
    val_episodes: int = 20  # INCREASED from 10
    test_episodes: int = 50  # INCREASED from 15 for robust evaluation
    seed: int = 42
    xai_samples: int = 10
    use_amp: bool = True
    label_smoothing: float = 0.1  # ADDED for better generalization
    early_stopping_patience: int = 20  # ADDED

CFG = Config()

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GPU optimization
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.cuda.empty_cache()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("Warning: GPU not available. Using CPU.")

# Create output directories
for d in ['splits', 'plots', 'xai', 'checkpoints', 'logs']:
    os.makedirs(os.path.join(CFG.output_dir, d), exist_ok=True)

# ================================================================================
# REPRODUCIBILITY
# ================================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CFG.seed)

# ================================================================================
# DATA HANDLING
# ================================================================================

def make_stratified_splits(config=CFG):
    """Create stratified train/val/test splits."""
    data = []
    root = Path(config.data_root)

    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {config.data_root}")

    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    for cls_name in classes:
        cls_path = root / cls_name
        images = [f for f in cls_path.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        for img_path in images:
            data.append({
                'image_path': str(img_path),
                'label': class_to_idx[cls_name],
                'class_name': cls_name
            })

    df = pd.DataFrame(data)
    print(f"Total samples: {len(df)}")

    X, y = df['image_path'].values, df['label'].values

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1-config.train_ratio), random_state=config.seed)
    train_idx, temp_idx = next(sss1.split(X, y))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=config.test_ratio/(config.val_ratio+config.test_ratio),
                                   random_state=config.seed)
    X_temp, y_temp = df_temp['image_path'].values, df_temp['label'].values
    val_idx, test_idx = next(sss2.split(X_temp, y_temp))

    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)

    df_train.to_csv(f"{config.output_dir}/splits/train.csv", index=False)
    df_val.to_csv(f"{config.output_dir}/splits/val.csv", index=False)
    df_test.to_csv(f"{config.output_dir}/splits/test.csv", index=False)

    print(f"Split sizes: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    return df_train, df_val, df_test


class FewShotDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        image = transforms.ToTensor()(image)
        if self.transform:
            image = self.transform(image)
        return image, int(row['label'])


class EpisodicSampler:
    """Sampler for N-way K-shot episodes."""
    def __init__(self, labels, n_way, k_shot, q_query, n_episodes, seed=42):
        self.labels = np.array(labels)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_episodes = n_episodes
        self.rng = np.random.RandomState(seed)

        self.class_indices = {}
        for label in np.unique(self.labels):
            self.class_indices[label] = np.where(self.labels == label)[0]

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            selected_classes = self.rng.choice(list(self.class_indices.keys()),
                                               size=self.n_way, replace=False)
            support_idx, query_idx = [], []
            for cls in selected_classes:
                n_samples = self.k_shot + self.q_query
                sampled = self.rng.choice(self.class_indices[cls], size=n_samples, replace=False)
                support_idx.extend(sampled[:self.k_shot])
                query_idx.extend(sampled[self.k_shot:])
            yield support_idx, query_idx


# Transforms - ENHANCED DATA AUGMENTATION FOR BETTER GENERALIZATION
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((CFG.img_size, CFG.img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),  # ADDED
    transforms.RandomRotation(30),  # INCREASED from 15
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # ADDED
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),  # ENHANCED
    transforms.RandomGrayscale(p=0.1),  # ADDED
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),  # ADDED
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # ADDED
    transforms.Normalize(mean=mean, std=std)
])

eval_transform = transforms.Compose([
    transforms.Resize((CFG.img_size, CFG.img_size)),
    transforms.Normalize(mean=mean, std=std)
])

# ================================================================================
# MODEL ARCHITECTURE
# ================================================================================

class ConvEncoder(nn.Module):
    """
    Deep CNN encoder for embedding images - OPTIMIZED FOR GPU UTILIZATION.

    IMPORTANT: Using inplace=False in all ReLU layers to avoid conflicts
    with backward hooks used in GradCAM.
    """
    def __init__(self, out_dim=512, dropout=0.3):  # Default out_dim increased to 512
        super().__init__()
        # Increased channels for better GPU utilization
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout/3),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout/3),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout/2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),  # Increased hidden size
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(1024, out_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class PrototypicalNetwork(nn.Module):
    """Prototypical Network for few-shot learning."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, support, support_labels, query):
        z_support = self.encoder(support)
        z_query = self.encoder(query)

        unique_labels = torch.unique(support_labels)
        prototypes = []
        for label in unique_labels:
            class_mask = (support_labels == label)
            prototype = z_support[class_mask].mean(dim=0)
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes)

        dists = self.euclidean_dist(z_query, prototypes)
        logits = -dists
        return logits, prototypes, z_query

    @staticmethod
    def euclidean_dist(x, y):
        n = x.size(0)
        m = y.size(0)
        xx = (x**2).sum(dim=1, keepdim=True).expand(n, m)
        yy = (y**2).sum(dim=1, keepdim=True).expand(m, n).t()
        dist = xx + yy - 2.0 * x @ y.t()
        return torch.clamp(dist, min=0.0)


# Initialize model
encoder = ConvEncoder(out_dim=CFG.embedding_dim)
model = PrototypicalNetwork(encoder).to(DEVICE)

print("="*60)
print("MODEL ARCHITECTURE")
print("="*60)
print(f"Device: {DEVICE}")
print(encoder)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"\nTrainable parameters: {trainable:,}")
print(f"Total parameters: {total:,}")
print("="*60)

# ================================================================================
# TRAINING
# ================================================================================

def prototypical_loss(logits, labels):
    """Cross-entropy loss with label smoothing for better generalization."""
    return F.cross_entropy(logits, labels, label_smoothing=CFG.label_smoothing)


class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def run_episode(model, optimizer, dataset, support_idx, query_idx, training=True, scaler=None):
    if training:
        model.train()
    else:
        model.eval()

    support_data = [dataset[i] for i in support_idx]
    query_data = [dataset[i] for i in query_idx]

    support_images = torch.stack([x[0] for x in support_data]).to(DEVICE)
    support_labels = torch.tensor([x[1] for x in support_data]).to(DEVICE)
    query_images = torch.stack([x[0] for x in query_data]).to(DEVICE)
    query_labels = torch.tensor([x[1] for x in query_data]).to(DEVICE)

    unique_labels = torch.unique(support_labels)
    label_map = {int(l): i for i, l in enumerate(unique_labels)}
    support_labels_mapped = torch.tensor([label_map[int(l)] for l in support_labels],
                                          dtype=torch.long, device=DEVICE)
    query_labels_mapped = torch.tensor([label_map[int(l)] for l in query_labels],
                                        dtype=torch.long, device=DEVICE)

    if training:
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits, _, _ = model(support_images, support_labels_mapped, query_images)
                loss = prototypical_loss(logits, query_labels_mapped)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _, _ = model(support_images, support_labels_mapped, query_images)
            loss = prototypical_loss(logits, query_labels_mapped)
            loss.backward()
            optimizer.step()
    else:
        with torch.no_grad():
            logits, _, _ = model(support_images, support_labels_mapped, query_images)
            loss = prototypical_loss(logits, query_labels_mapped)

    preds = torch.argmax(logits, dim=1)
    acc = (preds == query_labels_mapped).float().mean().item()

    return loss.item(), acc, preds.cpu().numpy(), query_labels_mapped.cpu().numpy()


# Create splits
df_train, df_val, df_test = make_stratified_splits()

# Datasets
train_dataset = FewShotDataset(df_train, transform=train_transform)
val_dataset = FewShotDataset(df_val, transform=eval_transform)

# Samplers
train_sampler = EpisodicSampler(df_train['label'].values, CFG.n_way, CFG.k_shot,
                                 CFG.q_query, CFG.episodes_per_epoch, seed=CFG.seed)
val_sampler = EpisodicSampler(df_val['label'].values, CFG.n_way, CFG.k_shot,
                               CFG.q_query_eval, CFG.val_episodes, seed=CFG.seed)

# Optimizer - AdamW with better regularization
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay,
                               betas=(0.9, 0.999), eps=1e-8)

# Cosine annealing with warm restarts for better convergence
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2, eta_min=1e-6
)

# Early stopping setup
early_stopping_patience = 20
early_stopping_counter = 0

# Mixed precision
scaler = torch.cuda.amp.GradScaler() if (CFG.use_amp and torch.cuda.is_available()) else None

# Training loop
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
best_val_acc = 0.0

print(f"\n{'='*60}")
print("TRAINING START")
print(f"{'='*60}")
print(f"Mixed Precision Training: {CFG.use_amp}")
print(f"Label Smoothing: {CFG.label_smoothing}")
print(f"Optimizer: AdamW (lr={CFG.lr}, wd={CFG.weight_decay})")
print(f"Scheduler: CosineAnnealingWarmRestarts")
print(f"Early Stopping Patience: {early_stopping_patience}")
print(f"{'='*60}\n")

total_start = time.time()

for epoch in range(1, CFG.n_epochs + 1):
    epoch_start = time.time()

    # Training
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    for support_idx, query_idx in train_sampler:
        loss, acc, _, _ = run_episode(model, optimizer, train_dataset,
                                       support_idx, query_idx, training=True, scaler=scaler)
        train_loss.update(loss)
        train_acc.update(acc)

    scheduler.step()

    # Validation
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    model.eval()
    for support_idx, query_idx in val_sampler:
        loss, acc, _, _ = run_episode(model, optimizer, val_dataset,
                                       support_idx, query_idx, training=False, scaler=None)
        val_loss.update(loss)
        val_acc.update(acc)

    # Record
    history['train_loss'].append(train_loss.avg)
    history['train_acc'].append(train_acc.avg)
    history['val_loss'].append(val_loss.avg)
    history['val_acc'].append(val_acc.avg)
    history['lr'].append(optimizer.param_groups[0]['lr'])

    epoch_time = time.time() - epoch_start

    print(f"Epoch {epoch:3d}/{CFG.n_epochs} | "
          f"Train Loss: {train_loss.avg:.4f} Acc: {train_acc.avg:.3f} | "
          f"Val Loss: {val_loss.avg:.4f} Acc: {val_acc.avg:.3f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
          f"Time: {epoch_time:.1f}s")

    # Save best with early stopping
    if val_acc.avg > best_val_acc:
        best_val_acc = val_acc.avg
        early_stopping_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': best_val_acc,
            'history': history
        }, f"{CFG.output_dir}/checkpoints/best_model.pth")
        print(f"  -> Saved new best model (val_acc: {best_val_acc:.4f})")
    else:
        early_stopping_counter += 1

    # Early stopping check
    if early_stopping_counter >= early_stopping_patience:
        print(f"\nEarly stopping triggered after {epoch} epochs")
        break

total_time = time.time() - total_start
print(f"\n{'='*60}")
print(f"Training complete!")
print(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
print(f"Best Val Acc: {best_val_acc:.4f}")
print(f"Epochs trained: {epoch}/{CFG.n_epochs}")
print(f"{'='*60}")

# ================================================================================
# EVALUATION
# ================================================================================

# Load best model
checkpoint = torch.load(f"{CFG.output_dir}/checkpoints/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])

def evaluate_model(model, df, split='test', n_runs=3):
    """Evaluate model with multiple runs for statistical significance."""
    all_run_metrics = []

    for run in range(n_runs):
        set_seed(CFG.seed + run)
        test_dataset = FewShotDataset(df, transform=eval_transform)
        test_sampler = EpisodicSampler(df['label'].values, CFG.n_way, CFG.k_shot,
                                        CFG.q_query_eval, CFG.test_episodes, seed=CFG.seed + run)

        all_preds, all_labels = [], []

        model.eval()
        for support_idx, query_idx in tqdm(test_sampler, desc=f"Evaluating {split} run {run+1}/{n_runs}"):
            _, _, preds, labels = run_episode(model, None, test_dataset,
                                                support_idx, query_idx, training=False, scaler=None)
            all_preds.extend(preds)
            all_labels.extend(labels)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        run_metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'f1_micro': f1_score(all_labels, all_preds, average='micro'),
            'precision_macro': precision_score(all_labels, all_preds, average='macro'),
            'recall_macro': recall_score(all_labels, all_preds, average='macro'),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        }
        all_run_metrics.append(run_metrics)

    # Average metrics across runs
    metrics = {
        'accuracy': np.mean([m['accuracy'] for m in all_run_metrics]),
        'accuracy_std': np.std([m['accuracy'] for m in all_run_metrics]),
        'f1_macro': np.mean([m['f1_macro'] for m in all_run_metrics]),
        'f1_macro_std': np.std([m['f1_macro'] for m in all_run_metrics]),
        'f1_micro': np.mean([m['f1_micro'] for m in all_run_metrics]),
        'precision_macro': np.mean([m['precision_macro'] for m in all_run_metrics]),
        'recall_macro': np.mean([m['recall_macro'] for m in all_run_metrics]),
        'confusion_matrix': all_run_metrics[-1]['confusion_matrix'],
    }

    return metrics

metrics = evaluate_model(model, df_test, 'test', n_runs=3)

with open(f"{CFG.output_dir}/test_metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n{'='*60}")
print("TEST SET RESULTS")
print(f"{'='*60}")
print(f"Accuracy:          {metrics['accuracy']:.4f} (±{metrics['accuracy_std']:.4f})")
print(f"F1-Score (Macro):  {metrics['f1_macro']:.4f} (±{metrics['f1_macro_std']:.4f})")
print(f"F1-Score (Micro):  {metrics['f1_micro']:.4f}")
print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
print(f"Recall (Macro):    {metrics['recall_macro']:.4f}")
print(f"{'='*60}")

# ================================================================================
# VISUALIZATIONS
# ================================================================================

# Training history
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
epochs = range(1, len(history['train_loss']) + 1)

axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Loss over Epochs')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Accuracy over Epochs')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

gap = [v - t for t, v in zip(history['train_acc'], history['val_acc'])]
axes[1, 1].bar(epochs, gap, color=['green' if g > 0 else 'red' for g in gap], alpha=0.7)
axes[1, 1].axhline(y=0, color='black', linestyle='--')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Val Acc - Train Acc')
axes[1, 1].set_title('Generalization Gap')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{CFG.output_dir}/plots/training_history.png", dpi=150)
plt.close()
print("Training history saved")

# Confusion Matrix
cm = np.array(metrics['confusion_matrix'])
class_names = sorted(df_train['class_name'].unique())

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

im1 = axes[0].imshow(cm, interpolation='nearest', cmap='Blues')
axes[0].figure.colorbar(im1, ax=axes[0])
axes[0].set_xticks(range(len(class_names)))
axes[0].set_yticks(range(len(class_names)))
axes[0].set_xticklabels(class_names, rotation=45, ha='right')
axes[0].set_yticklabels(class_names)
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')
axes[0].set_title('Confusion Matrix (Counts)')

for i in range(len(class_names)):
    for j in range(len(class_names)):
        axes[0].text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black')

cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
im2 = axes[1].imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
axes[1].figure.colorbar(im2, ax=axes[1])
axes[1].set_xticks(range(len(class_names)))
axes[1].set_yticks(range(len(class_names)))
axes[1].set_xticklabels(class_names, rotation=45, ha='right')
axes[1].set_yticklabels(class_names)
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')
axes[1].set_title('Confusion Matrix (Normalized)')

for i in range(len(class_names)):
    for j in range(len(class_names)):
        axes[1].text(j, i, f'{cm_norm[i, j]:.2f}', ha='center', va='center',
                    color='white' if cm_norm[i, j] > 0.5 else 'black')

plt.tight_layout()
plt.savefig(f"{CFG.output_dir}/plots/confusion_matrix.png", dpi=150)
plt.close()
print("Confusion matrix saved")

# ================================================================================
# XAI: EXPLAINABLE AI
# ================================================================================

class GradCAM:
    """
    Grad-CAM implementation.

    Note: The model must use inplace=False in ReLU layers for this to work
    correctly with backward hooks.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(self, input_tensor, support_images, support_labels, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        query_img = input_tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)
        logits, _, _ = self.model(support_images, support_labels, query_img)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        score = logits[0, target_class]
        score.backward()

        grads = self.gradients[0]
        acts = self.activations[0]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = (weights * acts).sum(dim=0)
        cam = F.relu(cam)
        cam = cam.cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (input_tensor.shape[1], input_tensor.shape[2]))
        return cam

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def compute_saliency_map(model, input_tensor, support_images, support_labels, target_class=None):
    model.eval()
    query_img = input_tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)
    logits, _, _ = model(support_images, support_labels, query_img)
    if target_class is None:
        target_class = logits.argmax(dim=1).item()
    score = logits[0, target_class]
    score.backward()
    saliency = query_img.grad.data.abs().squeeze().cpu().numpy()
    saliency = np.max(saliency, axis=0)
    if saliency.max() > 0:
        saliency = saliency / saliency.max()
    return saliency


def visualize_explanation(image, mask, alpha=0.5):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = image.cpu().numpy().transpose(1, 2, 0)
    img_denorm = np.clip(img_np * std + mean, 0, 1)
    cmap = plt.get_cmap('jet')
    heatmap = cmap(mask)[:, :, :3]
    overlay = np.clip((1 - alpha) * img_denorm + alpha * heatmap, 0, 1)
    return img_denorm, heatmap, overlay


print("\nGenerating XAI visualizations...")

test_dataset = FewShotDataset(df_test, transform=eval_transform)
sampler = EpisodicSampler(df_test['label'].values, CFG.n_way, CFG.k_shot,
                          CFG.q_query_eval, n_episodes=1, seed=CFG.seed)

support_idx, query_idx = next(iter(sampler))

support_data = [test_dataset[i] for i in support_idx]
support_images = torch.stack([x[0] for x in support_data]).to(DEVICE)
support_labels = torch.tensor([x[1] for x in support_data]).to(DEVICE)

unique_labels = torch.unique(support_labels)
label_map = {int(l): i for i, l in enumerate(unique_labels)}
support_labels_mapped = torch.tensor([label_map[int(l)] for l in support_labels],
                                       dtype=torch.long, device=DEVICE)

gradcam = GradCAM(model, model.encoder.encoder[6])

n_samples = min(5, len(query_idx))
for i in range(n_samples):
    query_img, true_label = test_dataset[query_idx[i]]

    with torch.no_grad():
        logits, _, _ = model(support_images, support_labels_mapped,
                            query_img.unsqueeze(0).to(DEVICE))
        pred_label = logits.argmax(dim=1).item()

    gradcam_mask = gradcam.generate(query_img, support_images, support_labels_mapped, pred_label)
    saliency_mask = compute_saliency_map(model, query_img, support_images, support_labels_mapped, pred_label)

    img_orig, heatmap_g, overlay_g = visualize_explanation(query_img, gradcam_mask)
    _, heatmap_s, overlay_s = visualize_explanation(query_img, saliency_mask)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    axes[0, 0].imshow(img_orig)
    axes[0, 0].set_title(f'Original\nTrue: {true_label}, Pred: {pred_label}')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(heatmap_g)
    axes[0, 1].set_title('Grad-CAM Heatmap')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(overlay_g)
    axes[0, 2].set_title('Grad-CAM Overlay')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(img_orig)
    axes[1, 0].set_title('Original')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(heatmap_s)
    axes[1, 1].set_title('Saliency Heatmap')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(overlay_s)
    axes[1, 2].set_title('Saliency Overlay')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f"{CFG.output_dir}/xai/sample_{i}_gradcam_saliency.png", dpi=150)
    plt.close()

gradcam.remove_hooks()
print(f"XAI visualizations saved to {CFG.output_dir}/xai/")
print("\nAll done!")
