"""
================================================================================
SIAMESE NETWORKS FOR FEW-SHOT IMAGE CLASSIFICATION WITH XAI
================================================================================
OPTIMIZED VERSION - Ready for Kaggle with GPU acceleration

Standard Few-Shot Setting (from diagram):
- 8-way 5-shot: 8 classes with 5 support examples each
- 40 support samples + 120 query samples per episode

Key Features:
- 128x128 image resolution for better quality
- 256-dimensional embeddings
- Mixed precision training
- Preloaded dataset in memory

Author: Research Implementation
Version: 3.0 (8-way 5-shot standard)
================================================================================
"""

import os
import sys
import random
from pathlib import Path
import json
import math
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import ttest_ind, sem

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 10

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    DATA_ROOT = '/kaggle/input/cucumber-dataset/Original Image'
    OUTPUT_DIR = '/kaggle/working'

    RNG_SEED = 42
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1

    # Few-shot parameters - ADAPTED for dataset with limited samples per class
    # Your dataset has 1031 training images across 8 classes = ~129 per class after split
    # 8-way 5-shot needs: 8 × (5 + 15) = 160 samples per episode
    N_WAY = 8           # 8 classes per episode
    K_SHOT = 3         # Reduced from 5 to ensure enough samples
    Q_QUERY = 10       # Reduced from 15 to match available samples

    EPOCHS = 30
    EPISODES_PER_EPOCH = 20
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MARGIN = 1.0

    # Image size for better quality
    IMAGE_SIZE = 128

    # Larger embedding dimension
    EMBEDDING_DIM = 256

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = Config()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def create_directories():
    dirs = {
        'SPLIT_DIR': os.path.join(cfg.OUTPUT_DIR, 'splits'),
        'PLOTS_DIR': os.path.join(cfg.OUTPUT_DIR, 'plots'),
        'XAI_DIR': os.path.join(cfg.OUTPUT_DIR, 'xai'),
        'CKPT_DIR': os.path.join(cfg.OUTPUT_DIR, 'checkpoints')
    }
    for name, path in dirs.items():
        os.makedirs(path, exist_ok=True)
        setattr(cfg, name, path)
    return dirs

# =============================================================================
# OPTIMIZED DATASET - PRELOADS ALL IMAGES INTO MEMORY
# =============================================================================

class CachedImageDataset(Dataset):
    """
    OPTIMIZED: Preloads ALL images into memory once.
    Uses 128x128 resolution for better image quality.
    """

    def __init__(self, df, transform=None, device='cuda'):
        self.df = df
        self.transform = transform
        self.device = device

        print(f"Preloading {len(df)} images into memory (128x128)...")
        self.images = []
        self.labels = []

        for idx in tqdm(range(len(df)), desc="Loading images"):
            row = self.df.iloc[idx]
            image = Image.open(row['image']).convert('RGB')

            # 128x128 for better quality
            image = image.resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), Image.BILINEAR)

            img_tensor = transforms.ToTensor()(image)
            img_tensor = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(img_tensor)

            self.images.append(img_tensor)
            self.labels.append(int(row['label']))

        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        print(f"Loaded {len(self)} images. Shape: {self.images.shape}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# =============================================================================
# DATA SPLITTING
# =============================================================================

def make_stratified_splits(root_dir):
    """Perform stratified train/val/test splitting."""
    global CLASS_NAMES

    data = []
    root = Path(root_dir)

    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    print(f"Discovered {len(classes)} classes: {classes}")
    CLASS_NAMES = classes

    if len(classes) < 2:
        raise ValueError('Dataset must contain at least 2 class subfolders')

    for lbl, cls in enumerate(classes):
        images = list((root / cls).glob('*'))
        images = [x for x in images if x.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']]
        for img in images:
            data.append({'image': str(img), 'label': lbl, 'class': cls})

    df = pd.DataFrame(data)
    print(f"Total images: {len(df)}")

    x, y = df['image'], df['label']

    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=(1-cfg.TRAIN_RATIO), random_state=cfg.RNG_SEED)
    train_idx, temp_idx = next(splitter1.split(x, y))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)

    test_ratio_adjusted = cfg.TEST_RATIO / (cfg.VAL_RATIO + cfg.TEST_RATIO)
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio_adjusted, random_state=cfg.RNG_SEED)
    val_idx, test_idx = next(splitter2.split(df_temp['image'], df_temp['label']))

    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)

    df_train.to_csv(os.path.join(cfg.SPLIT_DIR, 'train.csv'), index=False)
    df_val.to_csv(os.path.join(cfg.SPLIT_DIR, 'val.csv'), index=False)
    df_test.to_csv(os.path.join(cfg.SPLIT_DIR, 'test.csv'), index=False)

    print(f"\nStratified Split Results:")
    print(f"  Training:   {len(df_train)} images ({len(df_train)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(df_val)} images ({len(df_val)/len(df)*100:.1f}%)")
    print(f"  Testing:    {len(df_test)} images ({len(df_test)/len(df)*100:.1f}%)")

    return df_train, df_val, df_test, CLASS_NAMES


# =============================================================================
# EPISODIC SAMPLER
# =============================================================================

class SiameseSampler:
    """
    Adaptive sampler for few-shot episodes.
    Automatically adjusts N-way based on available samples per class.
    """

    def __init__(self, labels, n_way, k_shot, q_query, episodes, seed=None):
        if seed is None:
            seed = cfg.RNG_SEED

        self.labels = np.array(labels)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes = episodes
        self.rng = np.random.RandomState(seed)

        self.by_class = {c: np.where(self.labels == c)[0] for c in np.unique(self.labels)}

        # Calculate minimum required samples per class
        self.min_required = k_shot + q_query

        # Find valid classes (have enough samples)
        self.valid_classes = sorted([
            c for c in self.by_class.keys()
            if len(self.by_class[c]) >= k_shot
        ])

        if len(self.valid_classes) < 2:
            raise ValueError(
                f"Not enough samples in dataset. "
                f"Need at least {k_shot} samples per class for at least 2 classes."
            )

        print(f"SiameseSampler: Found {len(self.valid_classes)} valid classes "
              f"(need ≥{k_shot} samples each)")

    def __len__(self):
        return self.episodes

    def __iter__(self):
        for _ in range(self.episodes):
            # Adaptively select number of classes based on available samples
            max_classes = len(self.valid_classes)
            actual_n_way = min(self.n_way, max_classes)

            # Select classes
            selected_classes = self.rng.choice(
                self.valid_classes,
                size=actual_n_way,
                replace=False
            )

            support_idx = []
            query_idx = []

            for c in selected_classes:
                class_indices = self.by_class[c]
                available = len(class_indices)
                needed_support = self.k_shot
                needed_query = self.q_query

                # Calculate how many we can actually sample
                total_needed = needed_support + needed_query

                if available >= total_needed:
                    # Can sample without replacement
                    choices = self.rng.choice(class_indices, size=total_needed, replace=False)
                else:
                    # Need to be more creative
                    # First take all available for support
                    support_from_class = min(needed_support, available)

                    if support_from_class == available:
                        # All samples for support, can't use replacement for query
                        support_choices = class_indices.copy()
                        query_choices = np.array([])
                    else:
                        # Sample support without replacement
                        support_choices = self.rng.choice(class_indices, size=support_from_class, replace=False)
                        remaining_for_query = available - support_from_class

                        # How many query samples can we get without replacement?
                        query_without_replacement = min(needed_query, remaining_for_query)

                        if query_without_replacement > 0:
                            remaining_indices = [i for i in class_indices if i not in support_choices]
                            query_choices = self.rng.choice(remaining_indices, size=query_without_replacement, replace=False)
                        else:
                            query_choices = np.array([])

                        # If still need more query samples, use replacement
                        if len(query_choices) < needed_query:
                            extra_needed = needed_query - len(query_choices)
                            # Re-sample from support to get extra query samples
                            if len(support_choices) > 0:
                                extra = self.rng.choice(support_choices, size=extra_needed, replace=True)
                                query_choices = np.concatenate([query_choices, extra]) if len(query_choices) > 0 else extra

                    choices = np.concatenate([support_choices, query_choices])

                support_idx.extend(choices[:min(len(choices), needed_support)].tolist())

                # For query, take remaining or sample with replacement
                query_taken = min(len(choices) - needed_support, 0)
                remaining_for_query = choices[needed_support:] if len(choices) > needed_support else np.array([])

                if len(remaining_for_query) >= needed_query:
                    query_idx.extend(remaining_for_query[:needed_query].tolist())
                elif len(remaining_for_query) > 0:
                    # Take what we have, then use replacement
                    query_idx.extend(remaining_for_query.tolist())
                    deficit = needed_query - len(remaining_for_query)
                    if len(choices) > 0:
                        extra = self.rng.choice(choices[:needed_support], size=deficit, replace=True)
                        query_idx.extend(extra.tolist())
                else:
                    # No remaining samples, use replacement
                    if len(choices) > 0 and needed_query > 0:
                        query_choices = self.rng.choice(choices[:needed_support], size=needed_query, replace=True)
                        query_idx.extend(query_choices.tolist())

            yield support_idx, query_idx


# =============================================================================
# SIAMESE NETWORK MODEL (OPTIMIZED)
# =============================================================================

class SiameseEncoder(nn.Module):
    """
    ENHANCED: Larger encoder for 128x128 images with better representation.

    Architecture:
    - 4 convolutional blocks with progressive channel increase
    - Deeper network for richer feature extraction
    - 256-dimensional embedding for better discrimination
    """

    def __init__(self, out_dim=256):
        super().__init__()

        # Block 1: 3 -> 64 channels, output: 64x64
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Block 2: 64 -> 128 channels, output: 32x32
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Block 3: 128 -> 256 channels, output: 16x16
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Block 4: 256 -> 512 channels, output: 8x8
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Projection head with larger hidden dimension
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)

        embedding = self.fc(x)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class SiameseNetwork(nn.Module):
    """Siamese Network for few-shot classification with dynamic N-way support."""

    def __init__(self, encoder=None, embedding_dim=256, max_n_way=8):
        super().__init__()

        if encoder is None:
            self.encoder = SiameseEncoder(out_dim=embedding_dim)
        else:
            self.encoder = encoder

        self.embedding_dim = embedding_dim
        self.max_n_way = max_n_way

    def forward(self, support, support_labels, query):
        """Forward pass for few-shot classification."""
        support_emb = self.encoder(support)
        query_emb = self.encoder(query)

        n_query = query_emb.size(0)
        n_support = support_emb.size(0)

        # Get actual number of classes in this episode
        unique_labels = torch.unique(support_labels)
        n_way = len(unique_labels)

        # Compute pairwise distances
        diff = query_emb.unsqueeze(1) - support_emb.unsqueeze(0)
        dists = torch.sum(diff ** 2, dim=2)
        similarities = torch.exp(-dists)

        # Aggregate similarities by class
        logits = torch.zeros(n_query, n_way, device=query_emb.device)

        for idx, c in enumerate(unique_labels):
            mask = (support_labels == c).unsqueeze(0)
            class_sim = (similarities * mask.float()).sum(dim=1)
            logits[:, idx] = class_sim

        return logits, query_emb, support_emb


class PairwiseCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        return F.cross_entropy(logits, labels)


# =============================================================================
# OPTIMIZED TRAINING FUNCTIONS
# =============================================================================

def run_episode(model, optimizer, dataset, support_idx, query_idx, criterion, scaler):
    """OPTIMIZED episode training with mixed precision."""
    model.train()

    # Handle edge case: not enough samples
    if len(support_idx) < 2 or len(query_idx) < 1:
        return 0.0, 0.0

    support_idx_t = torch.tensor(support_idx, dtype=torch.long)
    query_idx_t = torch.tensor(query_idx, dtype=torch.long)

    support_images = dataset.images[support_idx_t].to(cfg.DEVICE)
    support_labels = dataset.labels[support_idx_t].to(cfg.DEVICE)
    query_images = dataset.images[query_idx_t].to(cfg.DEVICE)
    query_labels = dataset.labels[query_idx_t].to(cfg.DEVICE)

    unique = torch.unique(support_labels)

    # Edge case: only one class in support
    if len(unique) < 2:
        return 0.0, 0.0

    label_map = {int(c): i for i, c in enumerate(unique)}
    support_labels_mapped = torch.tensor(
        [label_map[int(l)] for l in support_labels],
        dtype=torch.long, device=cfg.DEVICE
    )
    query_labels_mapped = torch.tensor(
        [label_map[int(l)] for l in query_labels],
        dtype=torch.long, device=cfg.DEVICE
    )

    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        logits, _, _ = model(support_images, support_labels_mapped, query_images)

    # Handle edge case
    if logits.numel() == 0 or logits.shape[0] == 0:
        return 0.0, 0.0

    loss = criterion(logits, query_labels_mapped)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    preds = torch.argmax(logits, dim=1)
    acc = (preds == query_labels_mapped).float().mean().item()

    return loss.item(), acc


def validate_episode(model, dataset, support_idx, query_idx, criterion):
    """OPTIMIZED validation."""
    model.eval()

    # Handle edge case: not enough samples
    if len(support_idx) < 2 or len(query_idx) < 1:
        return 0.0, 0.0

    with torch.no_grad():
        support_idx_t = torch.tensor(support_idx, dtype=torch.long)
        query_idx_t = torch.tensor(query_idx, dtype=torch.long)

        support_images = dataset.images[support_idx_t].to(cfg.DEVICE)
        support_labels = dataset.labels[support_idx_t].to(cfg.DEVICE)
        query_images = dataset.images[query_idx_t].to(cfg.DEVICE)
        query_labels = dataset.labels[query_idx_t].to(cfg.DEVICE)

        unique = torch.unique(support_labels)

        # Edge case: only one class in support
        if len(unique) < 2:
            return 0.0, 0.0

        label_map = {int(c): i for i, c in enumerate(unique)}
        support_labels_mapped = torch.tensor(
            [label_map[int(l)] for l in support_labels],
            dtype=torch.long, device=cfg.DEVICE
        )
        query_labels_mapped = torch.tensor(
            [label_map[int(l)] for l in query_labels],
            dtype=torch.long, device=cfg.DEVICE
        )

        with torch.cuda.amp.autocast():
            logits, _, _ = model(support_images, support_labels_mapped, query_images)

        # Handle edge case: logits might be empty
        if logits.numel() == 0 or logits.shape[0] == 0:
            return 0.0, 0.0

        loss = criterion(logits, query_labels_mapped)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == query_labels_mapped).float().mean().item()

        return loss.item(), acc


def train_model(model, df_train, df_val, n_epochs=None, lr=None, verbose=True):
    """OPTIMIZED training with mixed precision and preloaded data."""
    global CLASS_NAMES

    if n_epochs is None: n_epochs = cfg.EPOCHS
    if lr is None: lr = cfg.LEARNING_RATE

    print("\nLoading datasets into memory...")
    train_dataset = CachedImageDataset(df_train, None, cfg.DEVICE)
    val_dataset = CachedImageDataset(df_val, None, cfg.DEVICE)

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")

    train_sampler = SiameseSampler(
        df_train['label'].to_numpy(),
        n_way=cfg.N_WAY, k_shot=cfg.K_SHOT, q_query=cfg.Q_QUERY,
        episodes=cfg.EPISODES_PER_EPOCH
    )
    val_sampler = SiameseSampler(
        df_val['label'].to_numpy(),
        n_way=cfg.N_WAY, k_shot=cfg.K_SHOT, q_query=cfg.Q_QUERY,
        episodes=10
    )

    criterion = PairwiseCrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.WEIGHT_DECAY)

    total_steps = n_epochs * cfg.EPISODES_PER_EPOCH
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=0.3, anneal_strategy='cos'
    )

    scaler = torch.cuda.amp.GradScaler()

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'epoch_times': []}
    best_val_acc = 0.0
    best_ckpt = None

    print(f"\n{'='*60}")
    print("TRAINING SIAMESE NETWORK (OPTIMIZED)")
    print(f"{'='*60}")
    print(f"Device: {cfg.DEVICE}")
    print(f"Epochs: {n_epochs}, Episodes/Epoch: {cfg.EPISODES_PER_EPOCH}")
    print(f"Image Size: {cfg.IMAGE_SIZE}x{cfg.IMAGE_SIZE}")
    print(f"Few-Shot Setting: {cfg.N_WAY}-way {cfg.K_SHOT}-shot")
    print(f"  → Support samples per episode: {cfg.N_WAY * cfg.K_SHOT} ({cfg.N_WAY} classes × {cfg.K_SHOT} samples)")
    print(f"  → Query samples per episode: {cfg.N_WAY * cfg.Q_QUERY} ({cfg.N_WAY} classes × {cfg.Q_QUERY} samples)")
    print(f"{'='*60}\n")

    epoch_start = time.time()
    valid_train_episodes = 0
    valid_val_episodes = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses, train_accs = [], []

        for support_idx, query_idx in train_sampler:
            loss, acc = run_episode(model, optimizer, train_dataset, support_idx, query_idx, criterion, scaler)
            if loss > 0 and acc > 0:  # Only count valid episodes
                train_losses.append(loss)
                train_accs.append(acc)
                valid_train_episodes += 1
            scheduler.step()

        model.eval()
        val_losses, val_accs = [], []

        with torch.no_grad():
            for support_idx, query_idx in val_sampler:
                loss, acc = validate_episode(model, val_dataset, support_idx, query_idx, criterion)
                if loss > 0 and acc > 0:
                    val_losses.append(loss)
                    val_accs.append(acc)
                    valid_val_episodes += 1

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        train_acc = float(np.mean(train_accs)) if train_accs else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_acc = float(np.mean(val_accs)) if val_accs else 0.0
        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)

        if verbose and epoch % 2 == 0:
            print(f"Epoch {epoch:3d}/{n_epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f} | Time: {epoch_time:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt = os.path.join(cfg.CKPT_DIR, 'best_siamese.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, best_ckpt)

        epoch_start = time.time()

    print(f"\nTraining Complete! Best Val Accuracy: {best_val_acc:.4f}")
    print(f"Valid training episodes: {valid_train_episodes}, Valid validation episodes: {valid_val_episodes}")

    return history, best_ckpt


# =============================================================================
# EVALUATION
# =============================================================================

def compute_ece(probs, labels, n_bins=15):
    """Compute Expected Calibration Error."""
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        in_bin = confidences.gt(bin_boundaries[i]) & confidences.le(bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece.item()


def evaluate_model(model, df_test, episodes=20):
    """OPTIMIZED evaluation."""
    global CLASS_NAMES

    print("\nLoading test dataset...")
    test_dataset = CachedImageDataset(df_test, None, cfg.DEVICE)
    test_sampler = SiameseSampler(
        df_test['label'].to_numpy(),
        n_way=cfg.N_WAY, k_shot=cfg.K_SHOT, q_query=cfg.Q_QUERY,
        episodes=episodes
    )

    criterion = PairwiseCrossEntropyLoss()
    model.eval()

    y_true, y_pred, y_prob = [], [], []
    all_loss, all_acc = [], []

    with torch.no_grad():
        for support_idx, query_idx in test_sampler:
            # Skip invalid episodes
            if len(support_idx) < 2 or len(query_idx) < 1:
                continue

            support_idx_t = torch.tensor(support_idx, dtype=torch.long)
            query_idx_t = torch.tensor(query_idx, dtype=torch.long)

            support_images = test_dataset.images[support_idx_t].to(cfg.DEVICE)
            support_labels = test_dataset.labels[support_idx_t].to(cfg.DEVICE)
            query_images = test_dataset.images[query_idx_t].to(cfg.DEVICE)
            query_labels = test_dataset.labels[query_idx_t].to(cfg.DEVICE)

            unique = torch.unique(support_labels)
            if len(unique) < 2:
                continue

            label_map = {int(c): i for i, c in enumerate(unique)}
            support_labels_mapped = torch.tensor(
                [label_map[int(l)] for l in support_labels],
                dtype=torch.long, device=cfg.DEVICE
            )
            query_labels_mapped = torch.tensor(
                [label_map[int(l)] for l in query_labels],
                dtype=torch.long, device=cfg.DEVICE
            )

            with torch.cuda.amp.autocast():
                logits, _, _ = model(support_images, support_labels_mapped, query_images)

            if logits.numel() == 0 or logits.shape[0] == 0:
                continue

            loss = criterion(logits, query_labels_mapped)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            probs = F.softmax(logits, dim=1).cpu().numpy()

            all_loss.append(loss.item())
            all_acc.append((preds == query_labels_mapped.cpu().numpy()).mean())

            y_true.extend(query_labels_mapped.cpu().numpy().tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())

    if len(y_true) == 0:
        return {'accuracy': 0.0, 'f1_macro': 0.0, 'f1_micro': 0.0, 'f1_weighted': 0.0,
                'precision_per_class': [], 'recall_per_class': [], 'f1_per_class': [],
                'ece': 0.0, 'confusion_matrix': [], 'test_loss': 0.0, 'test_acc': 0.0}

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    precision, recall, f1_per_class, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    ece_val = compute_ece(torch.from_numpy(np.array(y_prob)), torch.from_numpy(np.array(y_true)), n_bins=15)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'ece': ece_val,
        'confusion_matrix': cm.tolist(),
        'test_loss': float(np.mean(all_loss)),
        'test_acc': float(np.mean(all_acc))
    }

    return metrics


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_training_history(history, save_path=None):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix with proper class alignment."""
    global CLASS_NAMES

    # Handle shape mismatch
    if cm.shape[0] != cm.shape[1]:
        cm = confusion_matrix(y_true, y_pred)

    n_classes = cm.shape[0]

    # Use full class names if available, otherwise create generic ones
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(n_classes)]
    elif len(class_names) < n_classes:
        class_names = list(class_names) + [f'Class_{i}' for i in range(len(class_names), n_classes)]

    class_names = class_names[:n_classes]

    fig, ax = plt.subplots(figsize=(10, 8))

    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, vmin=0, vmax=1, cbar_kws={'label': 'Proportion'})

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix - Siamese Network', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_per_class_metrics(metrics, class_names, save_path=None):
    """Plot per-class metrics with proper class alignment."""
    global CLASS_NAMES

    # Get all unique classes from metrics
    n_classes_metrics = len(metrics.get('f1_per_class', []))

    # Use full class names if available, otherwise create generic ones
    if class_names is None:
        class_names = [f'Class_{i}' for i in range(n_classes_metrics)]
    elif len(class_names) < n_classes_metrics:
        # Extend class names if we have more classes in metrics
        class_names = list(class_names) + [f'Class_{i}' for i in range(len(class_names), n_classes_metrics)]

    # Truncate to match metrics
    class_names = class_names[:n_classes_metrics]
    x = np.arange(len(class_names))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    precision = metrics.get('precision_per_class', [0] * n_classes_metrics)
    recall = metrics.get('recall_per_class', [0] * n_classes_metrics)
    f1 = metrics.get('f1_per_class', [0] * n_classes_metrics)

    axes[0].bar(x, precision, width, label='Precision', color='steelblue')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Per-Class Precision')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c[:10] for c in class_names], rotation=45, ha='right')
    axes[0].set_ylim([0, 1.1])
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(x, recall, width, label='Recall', color='forestgreen')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Per-Class Recall')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c[:10] for c in class_names], rotation=45, ha='right')
    axes[1].set_ylim([0, 1.1])
    axes[1].grid(True, alpha=0.3, axis='y')

    axes[2].bar(x, f1, width, label='F1-Score', color='coral')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('Score')
    axes[2].set_title('Per-Class F1-Score')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([c[:10] for c in class_names], rotation=45, ha='right')
    axes[2].set_ylim([0, 1.1])
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


# =============================================================================
# XAI METHODS
# =============================================================================

def compute_saliency_map(model, input_tensor, support_images, support_labels, target_class=None):
    """Compute gradient-based saliency map."""
    model.eval()
    input_tensor = input_tensor.unsqueeze(0).clone().detach().requires_grad_(True)

    unique = torch.unique(support_labels)
    label_map = {int(c): i for i, c in enumerate(unique)}
    support_labels_mapped = torch.tensor(
        [label_map[int(l)] for l in support_labels],
        dtype=torch.long, device=cfg.DEVICE
    )

    logits, _, _ = model(support_images, support_labels_mapped, input_tensor)

    if target_class is None:
        target_class = torch.argmax(logits, dim=1).item()

    score = logits[0, target_class]
    score.backward()

    saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()
    saliency = np.max(saliency, axis=0)
    saliency = saliency - saliency.min()
    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    return saliency


def generate_xai(model, df_test, n_samples=5):
    """Generate XAI explanations."""
    global CLASS_NAMES

    print("\nGenerating XAI explanations...")
    ds = CachedImageDataset(df_test, None, cfg.DEVICE)
    sampler = SiameseSampler(
        df_test['label'].to_numpy(),
        n_way=cfg.N_WAY, k_shot=cfg.K_SHOT, q_query=cfg.Q_QUERY,
        episodes=1
    )

    support_idx, query_idx = next(iter(sampler))

    support_images = ds.images[torch.tensor(support_idx, dtype=torch.long)].to(cfg.DEVICE)
    support_labels = ds.labels[torch.tensor(support_idx, dtype=torch.long)].to(cfg.DEVICE)
    query_images = ds.images[torch.tensor(query_idx, dtype=torch.long)].to(cfg.DEVICE)
    query_labels = ds.labels[torch.tensor(query_idx, dtype=torch.long)].to(cfg.DEVICE)

    unique = torch.unique(support_labels)
    label_map = {int(c): i for i, c in enumerate(unique)}
    reverse_label_map = {v: k for k, v in label_map.items()}
    support_labels_mapped = torch.tensor(
        [label_map[int(l)] for l in support_labels],
        dtype=torch.long, device=cfg.DEVICE
    )

    print(f"Support: {len(support_idx)} images, Query: {len(query_idx)} images")

    results = []
    for idx in range(min(n_samples, len(query_idx))):
        img = query_images[idx]
        true_label = query_labels[idx].item()

        logits, _, _ = model(support_images, support_labels_mapped, img.unsqueeze(0))
        pred = torch.argmax(logits, dim=1).item()
        confidence = F.softmax(logits, dim=1)[0, pred].item()
        pred_original = reverse_label_map[pred]

        sal_map = compute_saliency_map(model, img, support_images, support_labels, pred)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        img_np = img.cpu().numpy().transpose(1, 2, 0)
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        img_np = np.clip(img_np * std + mean, 0, 1)

        axes[0].imshow(img_np)
        axes[0].set_title(f'Original Image\nTrue: {CLASS_NAMES[true_label]}')
        axes[0].axis('off')

        axes[1].imshow(sal_map, cmap='jet')
        axes[1].set_title('Saliency Map')
        axes[1].axis('off')

        axes[2].imshow(img_np)
        axes[2].imshow(sal_map, cmap='jet', alpha=0.6)
        axes[2].set_title(f'Overlay\nPred: {CLASS_NAMES[pred_original]}, Conf: {confidence:.2f}')
        axes[2].axis('off')

        plt.tight_layout()
        save_path = os.path.join(cfg.XAI_DIR, f'saliency_{idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        results.append({
            'sample': idx,
            'true_label': CLASS_NAMES[true_label] if true_label < len(CLASS_NAMES) else f'Class {true_label}',
            'pred_label': CLASS_NAMES[pred_original] if pred_original < len(CLASS_NAMES) else f'Class {pred_original}',
            'confidence': confidence,
            'sparsity': float(np.mean(sal_map < 0.6))
        })

        print(f"Sample {idx}: True={CLASS_NAMES[true_label]}, Pred={CLASS_NAMES[pred_original]}, Conf={confidence:.3f}")

    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

CLASS_NAMES = None

def main():
    """Execute complete Siamese Network pipeline."""
    global CLASS_NAMES

    print(f"\n{'='*70}")
    print(f"SIAMESE NETWORKS FOR FEW-SHOT LEARNING WITH XAI")
    print(f"{'='*70}\n")

    start_time = time.time()

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        print("WARNING: Running on CPU - will be SLOW!")
    print(f"Device: {cfg.DEVICE}")
    print(f"{'='*70}\n")

    print("="*50)
    print("STEP 1: DATA PREPARATION")
    print("="*50)
    df_train, df_val, df_test, CLASS_NAMES = make_stratified_splits(cfg.DATA_ROOT)

    # Save CLASS_NAMES for later use
    n_total_classes = len(CLASS_NAMES)
    print(f"\nTotal classes in dataset: {n_total_classes}")

    print("\n" + "="*50)
    print("STEP 2: MODEL TRAINING")
    print("="*50)

    model = SiameseNetwork(embedding_dim=cfg.EMBEDDING_DIM).to(cfg.DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    history, best_ckpt = train_model(
        model, df_train, df_val,
        n_epochs=cfg.EPOCHS,
        lr=cfg.LEARNING_RATE
    )

    plot_training_history(history, save_path=os.path.join(cfg.PLOTS_DIR, 'training_history.png'))

    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'siamese_final.pth'))
    with open(os.path.join(cfg.OUTPUT_DIR, 'train_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("\nLoading best model checkpoint...")
    checkpoint = torch.load(best_ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with val_acc: {checkpoint['val_acc']:.4f}")

    print("\n" + "="*50)
    print("STEP 3: MODEL EVALUATION")
    print("="*50)

    test_metrics = evaluate_model(model, df_test)

    print(f"\n{'='*60}")
    print("TEST SET METRICS")
    print(f"{'='*60}")
    print(f"Overall Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"F1-Score (Macro):  {test_metrics['f1_macro']:.4f}")
    print(f"F1-Score (Micro):  {test_metrics['f1_micro']:.4f}")
    print(f"F1-Score (Weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"ECE (Calibration): {test_metrics['ece']:.4f}")

    # Display per-class metrics using actual class count from metrics
    n_classes_in_metrics = len(test_metrics.get('f1_per_class', []))
    print(f"\nPer-Class F1 Scores ({n_classes_in_metrics} classes in metrics):")
    for i, f1 in enumerate(test_metrics['f1_per_class']):
        class_name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'Class {i}'
        print(f"  {class_name[:20]:20s}: {f1:.4f}")
    print(f"{'='*60}\n")

    with open(os.path.join(cfg.OUTPUT_DIR, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    print("Generating visualizations...")

    # Use class names matching the metrics
    metrics_class_names = [f'Class {i}' for i in range(n_classes_in_metrics)]

    plot_confusion_matrix(
        np.array(test_metrics['confusion_matrix']),
        metrics_class_names,
        save_path=os.path.join(cfg.PLOTS_DIR, 'confusion_matrix.png')
    )

    plot_per_class_metrics(
        test_metrics, metrics_class_names,
        save_path=os.path.join(cfg.PLOTS_DIR, 'per_class_metrics.png')
    )

    print("\n" + "="*50)
    print("STEP 4: XAI VISUALIZATION")
    print("="*50)

    xai_results = generate_xai(model, df_test, n_samples=5)

    with open(os.path.join(cfg.OUTPUT_DIR, 'xai_results.json'), 'w') as f:
        json.dump(xai_results, f, indent=2)

    total_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Total execution time: {total_time/60:.1f} minutes")
    print(f"\nFinal Results:")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test F1-Macro: {test_metrics['f1_macro']:.4f}")
    print(f"  ECE: {test_metrics['ece']:.4f}")
    print(f"\nOutputs saved to: {cfg.OUTPUT_DIR}")
    print(f"{'='*70}\n")

    return model, test_metrics, history


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    set_seed(cfg.RNG_SEED)
    create_directories()
    model, test_metrics, history = main()