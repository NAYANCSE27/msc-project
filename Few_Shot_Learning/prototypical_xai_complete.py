"""
================================================================================
Prototypical Networks with Explainable AI (XAI) for Few-Shot Learning
================================================================================

Research-grade implementation for Q1-level publication.

Dataset: 8 classes × 160 images = 1,280 total images
Split: 80% Train (1,024) / 10% Val (128) / 10% Test (128)
Task: N-way K-shot few-shot learning with Prototypical Networks

Author: Research Implementation
Date: 2026-04-04
Environment: Kaggle Notebook (GPU: Tesla T4/P100, 15GB limit)

================================================================================
MATHEMATICAL FORMULATIONS
================================================================================

1. PROTOTYPE COMPUTATION (Snell et al., 2017):
   -------------------------------------------------
   For each class c in the support set S_c, the prototype is:

   p_c = (1 / |S_c|) Σ_{(x_i, y_i) ∈ S_c} f_φ(x_i)

   where:
   - f_φ: embedding function (CNN encoder)
   - S_c: support set for class c
   - |S_c|: number of support samples (K-shot)

2. EUCLIDEAN DISTANCE METRIC:
   -------------------------------------------------
   The squared Euclidean distance between query embedding and prototype:

   d(z, p_c) = ||f_φ(z) - p_c||²_2 = Σ_j (f_φ(z)_j - p_{c,j})²

3. SOFTMAX OVER DISTANCES (Logits):
   -------------------------------------------------
   The classification score for class c:

   a_c(z) = -d(z, p_c) = -||f_φ(z) - p_c||²_2

   P(y = c | z) = softmax(a(z))_c = exp(a_c(z)) / Σ_{c'} exp(a_{c'}(z))

4. NEGATIVE LOG-LIKELIHOOD LOSS:
   -------------------------------------------------
   L(φ) = - (1 / |Q|) Σ_{(z_j, y_j) ∈ Q} log P(y = y_j | z_j)

   where Q is the query set.

5. EVALUATION METRICS:
   -------------------------------------------------
   Accuracy:  ACC = (1/N) Σ_i 1(ŷ_i = y_i)

   F1-Score:  F1 = 2 × (Precision × Recall) / (Precision + Recall)

   Expected Calibration Error:
   ECE = Σ_{m=1}^M (|B_m| / n) |acc(B_m) - conf(B_m)|

   Attribution Sparsity:
   S(A) = (1 / HW) Σ_{i,j} 1(|A_{i,j}| < τ)

================================================================================
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
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, models
from torchvision.utils import make_grid, save_image

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, precision_recall_fscore_support,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import StratifiedShuffleSplit
from scipy import stats
from scipy.stats import ttest_ind, wilcoxon
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Kaggle

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ================================================================================
# CONFIGURATION & HYPERPARAMETERS
# ================================================================================

@dataclass
class Config:
    """Centralized configuration for reproducibility."""
    # Paths
    data_root: str = '/kaggle/input/few-shot-data'  # Kaggle input path
    output_dir: str = '/kaggle/working'              # Kaggle output path

    # Dataset
    n_classes: int = 8
    n_images_per_class: int = 160
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    img_size: int = 128  # Reduced for memory efficiency

    # Few-shot settings
    n_way: int = 8       # Number of classes per episode
    k_shot: int = 5      # Support samples per class
    q_query: int = 15    # Query samples per class (training)
    q_query_eval: int = 5  # Query samples per class (val/test) - smaller due to limited samples

    # Training
    n_epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_step: int = 10
    lr_gamma: float = 0.5
    episodes_per_epoch: int = 20
    val_episodes: int = 10
    test_episodes: int = 15

    # Model
    embedding_dim: int = 128
    dropout: float = 0.3

    # Reproducibility
    seed: int = 42

    # XAI
    xai_samples: int = 10
    gradcam_layer: str = 'encoder.6'  # Target layer for Grad-CAM

    # Performance
    use_amp: bool = True  # Automatic Mixed Precision for faster training

    def __post_init__(self):
        """Create output directories."""
        self.split_dir = os.path.join(self.output_dir, 'splits')
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        self.xai_dir = os.path.join(self.output_dir, 'xai')
        self.ckpt_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')

        for d in [self.split_dir, self.plot_dir, self.xai_dir,
                   self.ckpt_dir, self.log_dir]:
            os.makedirs(d, exist_ok=True)

# Global configuration
CFG = Config()

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GPU optimization
def setup_gpu():
    """Configure GPU for optimal performance."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()  # Clear cache before start
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA: {torch.version.cuda}")
        print(f"✓ Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"✓ Initial GPU Memory: {torch.cuda.memory_allocated() / 1e6:.1f} MB allocated")
        return True
    else:
        print("⚠ Warning: GPU not available. Using CPU (will be very slow).")
        return False

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024**2

def print_model_details(model: nn.Module, config: Config = CFG):
    """Print detailed model information."""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    print(f"\nDevice: {DEVICE}")
    print(f"Mixed Precision: {config.use_amp}")
    print(f"\n{'='*60}")
    print("ENCODER ARCHITECTURE")
    print("="*60)
    print(model.encoder)
    print(f"\n{'='*60}")
    print("MODEL STATISTICS")
    print("="*60)
    trainable_params, total_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Model size:           {model_size:.2f} MB")
    print(f"\n{'='*60}")

    # Verify model is on GPU
    if DEVICE.type == 'cuda':
        print(f"Model device: {next(model.parameters()).device}")
        print(f"GPU Memory after model load: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print("="*60)

# ================================================================================
# REPRODUCIBILITY
# ================================================================================

def set_seed(seed: int = CFG.seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# ================================================================================
# DATA HANDLING & PREPROCESSING
# ================================================================================

def make_stratified_splits(config: Config = CFG) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits preserving class distribution.

    Mathematical guarantee:
    - Each split maintains the same class proportions as the original dataset
    - P(y=c | split) ≈ P(y=c) for all classes c

    Args:
        config: Configuration object with split ratios

    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames
    """
    print("\n" + "="*60)
    print("DATASET SPLITTING (Stratified)")
    print("="*60)

    data = []
    root = Path(config.data_root)

    if not root.exists():
        raise FileNotFoundError(f"Data root not found: {config.data_root}")

    # Discover classes
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if len(classes) != config.n_classes:
        raise ValueError(f"Expected {config.n_classes} classes, found {len(classes)}")

    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # Collect all images
    for cls_name in classes:
        cls_path = root / cls_name
        images = [f for f in cls_path.glob('*')
                 if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']]
        for img_path in images:
            data.append({
                'image_path': str(img_path),
                'label': class_to_idx[cls_name],
                'class_name': cls_name
            })

    df = pd.DataFrame(data)
    print(f"Total samples: {len(df)}")
    print(f"Classes: {classes}")
    print(f"\nClass distribution:")
    print(df['class_name'].value_counts().sort_index())

    # Stratified split: Train vs (Val + Test)
    X = df['image_path'].values
    y = df['label'].values

    # First split: Train (80%) vs Temp (20%)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - config.train_ratio),
                                   random_state=config.seed)
    train_idx, temp_idx = next(sss1.split(X, y))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)

    # Second split: Val (50% of temp) vs Test (50% of temp) → 10% each overall
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.test_ratio / (config.val_ratio + config.test_ratio),
        random_state=config.seed
    )
    X_temp = df_temp['image_path'].values
    y_temp = df_temp['label'].values
    val_idx, test_idx = next(sss2.split(X_temp, y_temp))

    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)

    # Save splits
    df_train.to_csv(os.path.join(config.split_dir, 'train.csv'), index=False)
    df_val.to_csv(os.path.join(config.split_dir, 'val.csv'), index=False)
    df_test.to_csv(os.path.join(config.split_dir, 'test.csv'), index=False)

    print(f"\n{'='*60}")
    print(f"Split sizes: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    print(f"{'='*60}\n")

    # Verify stratification
    print("Stratification check (class proportions):")
    for name, split_df in [('Train', df_train), ('Val', df_val), ('Test', df_test)]:
        props = split_df['label'].value_counts(normalize=True).sort_index()
        print(f"  {name}: {dict(props.round(3))}")

    return df_train, df_val, df_test


class FewShotDataset(Dataset):
    """Dataset for episodic few-shot learning."""

    def __init__(self, df: pd.DataFrame, transform: Optional[Callable] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

        # Pre-compute valid extensions
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]

        # Load and preprocess image
        img_path = row['image_path']
        image = Image.open(img_path).convert('RGB')
        image = transforms.ToTensor()(image)  # [0, 1] range

        if self.transform:
            image = self.transform(image)

        return image, int(row['label'])


class EpisodicSampler(Sampler):
    """
    Sampler for generating few-shot episodes.

    Each episode consists of:
    - Support set: n_way × k_shot samples
    - Query set: n_way × q_query samples

    Mathematical property:
    - Samples are drawn uniformly without replacement from each class
    - Classes are selected uniformly without replacement for each episode
    """

    def __init__(self, labels: np.ndarray, n_way: int, k_shot: int,
                 q_query: int, n_episodes: int, seed: int = 42):
        self.labels = np.array(labels)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_episodes = n_episodes
        self.rng = np.random.RandomState(seed)

        # Organize indices by class
        self.class_indices = {}
        unique_labels = np.unique(self.labels)
        for label in unique_labels:
            self.class_indices[label] = np.where(self.labels == label)[0]

        # Validate
        min_samples = k_shot + q_query
        for label, indices in self.class_indices.items():
            if len(indices) < min_samples:
                raise ValueError(
                    f"Class {label} has {len(indices)} samples, "
                    f"need at least {min_samples} for {k_shot}-shot, {q_query}-query"
                )

    def __len__(self) -> int:
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            # Sample n_way classes
            selected_classes = self.rng.choice(
                list(self.class_indices.keys()),
                size=self.n_way,
                replace=False
            )

            support_indices = []
            query_indices = []

            for cls in selected_classes:
                # Sample (k_shot + q_query) indices for this class
                n_samples = self.k_shot + self.q_query
                sampled = self.rng.choice(
                    self.class_indices[cls],
                    size=n_samples,
                    replace=False
                )

                support_indices.extend(sampled[:self.k_shot])
                query_indices.extend(sampled[self.k_shot:])

            yield support_indices, query_indices


def get_transforms(config: Config = CFG) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Create training and evaluation transforms.

    Training augmentation:
    - Random horizontal flip (data augmentation)
    - Random rotation (±15 degrees)
    - Color jitter (brightness, contrast, saturation)
    - Normalization with ImageNet statistics

    Evaluation:
    - Resize and normalize only
    """
    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
        transforms.Normalize(mean=mean, std=std)
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, eval_transform


# ================================================================================
# MODEL ARCHITECTURE: PROTOTYPICAL NETWORKS
# ================================================================================

class ConvEncoder(nn.Module):
    """
    Convolutional encoder for embedding images into a feature space.

    Architecture:
    - ConvBlock: Conv → BatchNorm → ReLU → MaxPool
    - Progressive feature expansion: 32 → 64 → 128 → embedding_dim
    - Global average pooling for fixed-size output

    Mathematical transformation:
    f_φ: R^{C×H×W} → R^D where D = embedding_dim
    """

    def __init__(self, out_dim: int = CFG.embedding_dim, dropout: float = CFG.dropout):
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1: 128×128 → 64×64
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout / 2),

            # Block 2: 64×64 → 32×32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout / 2),

            # Block 3: 32×32 → 16×16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),

            # Block 4: 16×16 → 8×8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """He initialization for ReLU activation."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [B, 3, H, W]

        Returns:
            Embedding [B, out_dim]
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # L2 normalization for stable distance computation
        x = F.normalize(x, p=2, dim=1)

        return x


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network for Few-Shot Learning.

    Reference: Snell et al., "Prototypical Networks for Few-shot Learning", NIPS 2017

    Key concept:
    - Learn an embedding space where samples cluster by class
    - Classify by computing distances to class prototypes (centroids)
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, support: torch.Tensor, support_labels: torch.Tensor,
                query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass computing prototypes and distances.

        Args:
            support: Support images [N×K, C, H, W]
            support_labels: Support labels [N×K]
            query: Query images [N×Q, C, H, W]

        Returns:
            logits: Classification scores [N×Q, N]
            prototypes: Class prototypes [N, D]
            query_embeddings: Query embeddings [N×Q, D]
        """
        # Embed support and query
        z_support = self.encoder(support)   # [N×K, D]
        z_query = self.encoder(query)       # [N×Q, D]

        # Compute prototypes: p_c = mean(embeddings in class c)
        unique_labels = torch.unique(support_labels)
        prototypes = []

        for label in unique_labels:
            class_mask = (support_labels == label)
            class_embeddings = z_support[class_mask]
            prototype = class_embeddings.mean(dim=0)  # [D]
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)  # [N, D]

        # Compute squared Euclidean distances
        # dists[i,j] = ||query_i - prototype_j||²
        dists = self.euclidean_dist(z_query, prototypes)

        # Logits = -distances (closer = higher score)
        logits = -dists

        return logits, prototypes, z_query

    @staticmethod
    def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute squared Euclidean distance between x and y.

        Efficient computation:
        ||x - y||² = ||x||² + ||y||² - 2<x,y>

        Args:
            x: [n, d]
            y: [m, d]

        Returns:
            distances: [n, m]
        """
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)

        assert d == y.size(1), "Dimension mismatch"

        # ||x||²: [n, 1]
        x_norm = (x ** 2).sum(dim=1, keepdim=True)

        # ||y||²: [m, 1]
        y_norm = (y ** 2).sum(dim=1, keepdim=True)

        # x @ y.T: [n, m]
        xy = torch.mm(x, y.t())

        # ||x||² + ||y||² - 2<x,y>
        dist = x_norm.expand(n, m) + y_norm.t().expand(n, m) - 2.0 * xy

        # Numerical stability
        dist = torch.clamp(dist, min=0.0)

        return dist


# ================================================================================
# LOSS FUNCTION & TRAINING UTILITIES
# ================================================================================

def prototypical_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss for prototypical networks.

    L = - (1/|Q|) Σ log P(y_i | x_i)

    where P(y_i | x_i) = softmax(logits)_label
    """
    return F.cross_entropy(logits, labels)


class AverageMeter:
    """Track running averages for metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def run_episode(model: nn.Module, optimizer: torch.optim.Optimizer,
                dataset: Dataset, support_idx: List[int], query_idx: List[int],
                training: bool = True, scaler: Optional[torch.cuda.amp.GradScaler] = None) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a single training or evaluation episode.

    Args:
        model: PrototypicalNetwork
        optimizer: Optimizer (only used if training)
        dataset: Dataset
        support_idx: Support set indices
        query_idx: Query set indices
        training: Whether in training mode

    Returns:
        loss, accuracy, predictions, true_labels, probabilities
    """
    if training:
        model.train()
    else:
        model.eval()

    # Load data
    support_data = [dataset[i] for i in support_idx]
    query_data = [dataset[i] for i in query_idx]

    support_images = torch.stack([x[0] for x in support_data]).to(DEVICE)
    support_labels = torch.tensor([x[1] for x in support_data]).to(DEVICE)
    query_images = torch.stack([x[0] for x in query_data]).to(DEVICE)
    query_labels = torch.tensor([x[1] for x in query_data]).to(DEVICE)

    # Remap labels to 0..n_way-1 for this episode
    unique_labels = torch.unique(support_labels)
    label_map = {int(l): i for i, l in enumerate(unique_labels)}

    support_labels_mapped = torch.tensor(
        [label_map[int(l)] for l in support_labels],
        dtype=torch.long, device=DEVICE
    )
    query_labels_mapped = torch.tensor(
        [label_map[int(l)] for l in query_labels],
        dtype=torch.long, device=DEVICE
    )

    # Forward pass with optional AMP
    if training:
        optimizer.zero_grad()

        if scaler is not None:
            # Automatic Mixed Precision training
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

    # Compute predictions
    probs = F.softmax(logits, dim=1)
    preds = torch.argmax(logits, dim=1)

    acc = (preds == query_labels_mapped).float().mean().item()

    return (
        loss.item(),
        acc,
        preds.cpu().numpy(),
        query_labels_mapped.cpu().numpy(),
        probs.detach().cpu().numpy()
    )


# ================================================================================
# EVALUATION METRICS
# ================================================================================

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).

    ECE = Σ_m (|B_m|/n) |acc(B_m) - conf(B_m)|

    where B_m is the set of samples in bin m.

    Args:
        probs: Predicted probabilities [N, C]
        labels: True labels [N]
        n_bins: Number of bins

    Returns:
        ECE value
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def compute_attribution_sparsity(attribution: np.ndarray, threshold: float = 0.5) -> float:
    """
    Attribution Sparsity: Fraction of pixels with small attribution values.

    S(A) = (1/HW) Σ_{i,j} 1(|A_{i,j}| < τ)

    Higher sparsity = more focused explanations.

    Args:
        attribution: Attribution map [H, W]
        threshold: Threshold for considering a pixel as unimportant

    Returns:
        Sparsity score in [0, 1]
    """
    attr = np.abs(attribution)
    if attr.max() > 0:
        attr = attr / attr.max()

    sparsity = np.mean(attr < threshold)
    return float(sparsity)


def compute_brier_score(probs: np.ndarray, labels: np.ndarray, n_classes: int = CFG.n_classes) -> float:
    """
    Brier score for multi-class classification.

    BS = (1/N) Σ_i Σ_c (P(y_i = c) - 1(y_i = c))²

    Lower is better (perfect = 0).
    """
    n_samples = len(labels)
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), labels] = 1

    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return float(brier)


def evaluate_model(model: nn.Module, df: pd.DataFrame,
                   config: Config = CFG, split: str = 'test') -> Dict:
    """
    Comprehensive model evaluation.

    Returns dictionary with:
    - accuracy, f1_macro, f1_micro, f1_weighted
    - precision_per_class, recall_per_class, f1_per_class
    - confusion_matrix, ece, brier_score
    """
    _, eval_transform = get_transforms(config)
    dataset = FewShotDataset(df, transform=eval_transform)
    sampler = EpisodicSampler(
        df['label'].values,
        config.n_way, config.k_shot, config.q_query_eval,
        config.test_episodes, seed=config.seed
    )

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    all_losses = []

    print(f"\nEvaluating on {split} set...")
    for support_idx, query_idx in tqdm(sampler, desc=f"{split} episodes"):
        loss, acc, preds, labels, probs = run_episode(
            model, None, dataset, support_idx, query_idx, training=False
        )

        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs.extend(probs)
        all_losses.append(loss)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_micro': f1_score(all_labels, all_preds, average='micro'),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'precision_per_class': precision_score(all_labels, all_preds, average=None, zero_division=0).tolist(),
        'recall_per_class': recall_score(all_labels, all_preds, average=None, zero_division=0).tolist(),
        'f1_per_class': f1_score(all_labels, all_preds, average=None, zero_division=0).tolist(),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        'ece': compute_ece(all_probs, all_labels),
        'brier_score': compute_brier_score(all_probs, all_labels),
        'avg_loss': np.mean(all_losses)
    }

    return metrics


# ================================================================================
# XAI: EXPLAINABLE AI METHODS
# ================================================================================

class IntegratedGradients:
    """
    Integrated Gradients for attribution.

    Reference: Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017

    A(x) = (x - x_baseline) ⊙ ∫_0^1 ∇F(x_baseline + α(x - x_baseline)) dα

    where ⊙ denotes element-wise product.
    """

    def __init__(self, model: nn.Module, n_steps: int = 50):
        self.model = model
        self.n_steps = n_steps

    def attribute(self, input_tensor: torch.Tensor, support_images: torch.Tensor,
                  support_labels: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Compute integrated gradients attribution.

        Args:
            input_tensor: Query image [C, H, W]
            support_images: Support set images
            support_labels: Support set labels
            target_class: Target class (default: predicted)

        Returns:
            Attribution map [H, W]
        """
        self.model.eval()

        # Baseline: black image
        baseline = torch.zeros_like(input_tensor).to(DEVICE)
        query_img = input_tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)

        # Get prediction if target not specified
        with torch.no_grad():
            logits, _, _ = self.model(support_images, support_labels, query_img)
            if target_class is None:
                target_class = logits.argmax(dim=1).item()

        # Compute path integral
        total_grad = torch.zeros_like(query_img)

        for i in range(self.n_steps + 1):
            alpha = i / self.n_steps
            interpolated = baseline + alpha * (query_img - baseline)
            interpolated.requires_grad_(True)

            logits, _, _ = self.model(support_images, support_labels, interpolated)
            score = logits[0, target_class]

            grad = torch.autograd.grad(score, interpolated, create_graph=False)[0]
            total_grad += grad

        # Average and multiply by input
        avg_grad = total_grad / (self.n_steps + 1)
        integrated_grad = (query_img - baseline) * avg_grad

        # Sum over channels
        attribution = integrated_grad.squeeze().abs().sum(dim=0).cpu().numpy()

        # Normalize
        if attribution.max() > 0:
            attribution = attribution / attribution.max()

        return attribution


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017

    L^c_{Grad-CAM} = ReLU(Σ_k α^c_k A^k)

    where α^c_k = (1/Z) Σ_i Σ_j (∂y^c/∂A^k_{ij})
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
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

    def generate(self, input_tensor: torch.Tensor, support_images: torch.Tensor,
                 support_labels: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Query image [C, H, W]
            support_images: Support set
            support_labels: Support labels
            target_class: Target class

        Returns:
            Heatmap [H, W] in [0, 1]
        """
        self.model.eval()
        self.model.zero_grad()

        query_img = input_tensor.unsqueeze(0).to(DEVICE)
        query_img.requires_grad_(True)

        # Forward pass
        logits, _, _ = self.model(support_images, support_labels, query_img)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        # Backward pass
        score = logits[0, target_class]
        score.backward()

        # Compute Grad-CAM
        grads = self.gradients[0]  # [C, H, W]
        acts = self.activations[0]  # [C, H, W]

        # Global average pooling of gradients
        weights = grads.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]

        # Weighted combination of activation maps
        cam = (weights * acts).sum(dim=0)  # [H, W]
        cam = F.relu(cam)

        cam = cam.cpu().numpy()

        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input size
        cam = cv2.resize(cam, (input_tensor.shape[1], input_tensor.shape[2]))

        return cam

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def compute_saliency_map(model: nn.Module, input_tensor: torch.Tensor,
                         support_images: torch.Tensor, support_labels: torch.Tensor,
                         target_class: int = None) -> np.ndarray:
    """
    Vanilla gradient saliency map.

    S(x) = |∇_x F_y(x)|

    Reference: Simonyan et al., "Deep Inside Convolutional Networks", 2013
    """
    model.eval()

    query_img = input_tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)

    logits, _, _ = model(support_images, support_labels, query_img)

    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    score = logits[0, target_class]
    score.backward()

    saliency = query_img.grad.data.abs().squeeze().cpu().numpy()
    saliency = np.max(saliency, axis=0)  # Max over channels

    # Normalize
    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    return saliency


def visualize_explanation(image: torch.Tensor, mask: np.ndarray,
                          save_path: str, alpha: float = 0.5,
                          title: str = None):
    """
    Overlay attribution map on image.

    Args:
        image: Image tensor [3, H, W]
        mask: Attribution map [H, W] in [0, 1]
        save_path: Output path
        alpha: Overlay transparency
        title: Optional title
    """
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img_np = image.cpu().numpy().transpose(1, 2, 0)
    img_denorm = np.clip(img_np * std + mean, 0, 1)

    # Apply colormap to mask
    cmap = plt.get_cmap('jet')
    heatmap = cmap(mask)[:, :, :3]

    # Overlay
    overlay = np.clip((1 - alpha) * img_denorm + alpha * heatmap, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_denorm)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(heatmap)
    axes[1].set_title('Attribution Map')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    if title:
        plt.suptitle(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_xai_visualizations(model: nn.Module, df: pd.DataFrame,
                                  config: Config = CFG, n_samples: int = 10):
    """
    Generate XAI visualizations for sample images.
    """
    print("\n" + "="*60)
    print("GENERATING XAI VISUALIZATIONS")
    print("="*60)

    _, eval_transform = get_transforms(config)
    dataset = FewShotDataset(df, transform=eval_transform)

    # Get one episode
    sampler = EpisodicSampler(
        df['label'].values,
        config.n_way, config.k_shot, config.q_query_eval,
        n_episodes=1, seed=config.seed
    )

    support_idx, query_idx = next(iter(sampler))

    # Load support set
    support_data = [dataset[i] for i in support_idx]
    support_images = torch.stack([x[0] for x in support_data]).to(DEVICE)
    support_labels = torch.tensor([x[1] for x in support_data]).to(DEVICE)

    # Remap labels
    unique_labels = torch.unique(support_labels)
    label_map = {int(l): i for i, l in enumerate(unique_labels)}
    support_labels_mapped = torch.tensor(
        [label_map[int(l)] for l in support_labels],
        dtype=torch.long, device=DEVICE
    )

    # Initialize XAI methods
    gradcam = GradCAM(model, model.encoder.encoder[6])  # After 3rd conv block
    ig = IntegratedGradients(model, n_steps=30)

    sparsity_scores = {'gradcam': [], 'saliency': [], 'ig': []}

    n_samples = min(n_samples, len(query_idx))

    for i in tqdm(range(n_samples), desc="XAI samples"):
        query_data = dataset[query_idx[i]]
        query_img, true_label = query_data

        # Get prediction
        with torch.no_grad():
            logits, _, _ = model(
                support_images, support_labels_mapped,
                query_img.unsqueeze(0).to(DEVICE)
            )
            pred_label = logits.argmax(dim=1).item()

        # Generate explanations
        gradcam_mask = gradcam.generate(
            query_img, support_images, support_labels_mapped, target_class=pred_label
        )
        saliency_mask = compute_saliency_map(
            model, query_img, support_images, support_labels_mapped, target_class=pred_label
        )
        ig_mask = ig.attribute(
            query_img, support_images, support_labels_mapped, target_class=pred_label
        )

        # Compute sparsity
        sparsity_scores['gradcam'].append(compute_attribution_sparsity(gradcam_mask))
        sparsity_scores['saliency'].append(compute_attribution_sparsity(saliency_mask))
        sparsity_scores['ig'].append(compute_attribution_sparsity(ig_mask))

        # Save visualizations
        base_name = f"sample{i}_true{label_map.get(true_label, true_label)}_pred{pred_label}"

        visualize_explanation(
            query_img, gradcam_mask,
            os.path.join(config.xai_dir, f'{base_name}_gradcam.png'),
            title=f'{base_name} - Grad-CAM'
        )
        visualize_explanation(
            query_img, saliency_mask,
            os.path.join(config.xai_dir, f'{base_name}_saliency.png'),
            title=f'{base_name} - Saliency'
        )
        visualize_explanation(
            query_img, ig_mask,
            os.path.join(config.xai_dir, f'{base_name}_ig.png'),
            title=f'{base_name} - Integrated Gradients'
        )

    gradcam.remove_hooks()

    # Print sparsity summary
    print("\nAttribution Sparsity Summary:")
    for method, scores in sparsity_scores.items():
        print(f"  {method}: {np.mean(scores):.3f} ± {np.std(scores):.3f}")

    return sparsity_scores


# ================================================================================
# VISUALIZATION
# ================================================================================

def plot_training_history(history: Dict, save_path: str):
    """Plot training and validation metrics over epochs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss over Epochs')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy over Epochs')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning curves (smoothed)
    if len(history['train_loss']) > 5:
        from scipy.ndimage import uniform_filter1d
        train_smooth = uniform_filter1d(history['train_loss'], size=3)
        val_smooth = uniform_filter1d(history['val_loss'], size=3)
        axes[1, 0].plot(epochs, train_smooth, 'b-', label='Train (smoothed)', linewidth=2)
        axes[1, 0].plot(epochs, val_smooth, 'r-', label='Val (smoothed)', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss (smoothed)')
        axes[1, 0].set_title('Learning Curves (Smoothed)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Insufficient data\nfor smoothing',
                       ha='center', va='center', fontsize=12)
        axes[1, 0].set_title('Learning Curves')

    # Accuracy gap
    gap = [v - t for t, v in zip(history['train_acc'], history['val_acc'])]
    axes[1, 1].bar(epochs, gap, color=['green' if g > 0 else 'red' for g in gap], alpha=0.7)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Val Acc - Train Acc')
    axes[1, 1].set_title('Generalization Gap')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training history to {save_path}")


def plot_confusion_matrix(cm: List[List[int]], class_names: List[str], save_path: str):
    """Plot confusion matrix with percentages."""
    cm_arr = np.array(cm)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    im1 = axes[0].imshow(cm_arr, interpolation='nearest', cmap='Blues')
    axes[0].figure.colorbar(im1, ax=axes[0])
    axes[0].set_xticks(np.arange(len(class_names)))
    axes[0].set_yticks(np.arange(len(class_names)))
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].set_yticklabels(class_names)
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_title('Confusion Matrix (Counts)')

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            axes[0].text(j, i, format(cm_arr[i, j], 'd'),
                        ha='center', va='center',
                        color='white' if cm_arr[i, j] > cm_arr.max() / 2 else 'black')

    # Normalized (row percentages)
    cm_norm = cm_arr.astype('float') / cm_arr.sum(axis=1)[:, np.newaxis]
    im2 = axes[1].imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    axes[1].figure.colorbar(im2, ax=axes[1])
    axes[1].set_xticks(np.arange(len(class_names)))
    axes[1].set_yticks(np.arange(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].set_yticklabels(class_names)
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_title('Confusion Matrix (Normalized)')

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            axes[1].text(j, i, f'{cm_norm[i, j]:.2f}',
                        ha='center', va='center',
                        color='white' if cm_norm[i, j] > 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix to {save_path}")


def plot_per_class_metrics(metrics: Dict, class_names: List[str], save_path: str):
    """Plot per-class precision, recall, and F1-score."""
    precision = metrics['precision_per_class']
    recall = metrics['recall_per_class']
    f1 = metrics['f1_per_class']

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved per-class metrics to {save_path}")


def plot_calibration_curve(probs: np.ndarray, labels: np.ndarray,
                           n_classes: int, save_path: str, n_bins: int = 10):
    """Plot reliability diagram for calibration."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_confs = []
    bin_accs = []
    bin_counts = []

    for lower, upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > lower) & (confidences <= upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            bin_confs.append(avg_confidence_in_bin)
            bin_accs.append(accuracy_in_bin)
            bin_counts.append(np.sum(in_bin))
        else:
            bin_confs.append((lower + upper) / 2)
            bin_accs.append(0)
            bin_counts.append(0)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

    # Actual calibration
    ax.plot(bin_confs, bin_accs, 'ro-', label='Model Calibration', linewidth=2, markersize=8)

    # Gap bars
    for i, (conf, acc, count) in enumerate(zip(bin_confs, bin_accs, bin_counts)):
        if count > 0:
            ax.bar(conf, acc - conf, width=0.08, bottom=conf,
                  alpha=0.3, color='red' if acc < conf else 'green')

    ax.set_xlabel('Mean Predicted Confidence')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Reliability Diagram (Calibration Curve)')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved calibration curve to {save_path}")


# ================================================================================
# TRAINING PIPELINE
# ================================================================================

def train_model(model: nn.Module, df_train: pd.DataFrame, df_val: pd.DataFrame,
                config: Config = CFG) -> Tuple[Dict, str]:
    """
    Train the prototypical network.

    Returns:
        history: Dictionary of training metrics
        best_ckpt_path: Path to best model checkpoint
    """
    print("\n" + "="*60)
    print("TRAINING PROTOTYPICAL NETWORK")
    print("="*60)

    # Setup data
    train_transform, val_transform = get_transforms(config)
    train_dataset = FewShotDataset(df_train, transform=train_transform)
    val_dataset = FewShotDataset(df_val, transform=val_transform)

    train_sampler = EpisodicSampler(
        df_train['label'].values,
        config.n_way, config.k_shot, config.q_query,
        config.episodes_per_epoch, seed=config.seed
    )
    val_sampler = EpisodicSampler(
        df_val['label'].values,
        config.n_way, config.k_shot, config.q_query_eval,
        config.val_episodes, seed=config.seed
    )

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_step,
        gamma=config.lr_gamma
    )

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(config.ckpt_dir, 'best_model.pth')

    print(f"\nConfiguration:")
    print(f"  Epochs: {config.n_epochs}")
    print(f"  Episodes/epoch: {config.episodes_per_epoch}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  Device: {DEVICE}")
    print(f"  Mixed Precision: {config.use_amp}")
    if torch.cuda.is_available():
        print(f"  Initial GPU Memory: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"\n{'='*60}\n")

    # Initialize gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if (config.use_amp and torch.cuda.is_available()) else None

    start_time = time.time()

    for epoch in range(1, config.n_epochs + 1):
        epoch_start = time.time()

        # Training
        train_loss_meter = AverageMeter()
        train_acc_meter = AverageMeter()

        for support_idx, query_idx in train_sampler:
            loss, acc, _, _, _ = run_episode(
                model, optimizer, train_dataset,
                support_idx, query_idx, training=True, scaler=scaler
            )
            train_loss_meter.update(loss)
            train_acc_meter.update(acc)

        scheduler.step()

        # Validation
        val_loss_meter = AverageMeter()
        val_acc_meter = AverageMeter()

        model.eval()
        with torch.no_grad():
            for support_idx, query_idx in val_sampler:
                loss, acc, _, _, _ = run_episode(
                    model, optimizer, val_dataset,
                    support_idx, query_idx, training=False
                )
                val_loss_meter.update(loss)
                val_acc_meter.update(acc)

        # Record metrics
        history['train_loss'].append(train_loss_meter.avg)
        history['train_acc'].append(train_acc_meter.avg)
        history['val_loss'].append(val_loss_meter.avg)
        history['val_acc'].append(val_acc_meter.avg)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        epoch_time = time.time() - epoch_start

        # Print progress with GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1e6
            mem_info = f" | GPU: {gpu_mem:.1f} MB"
        else:
            mem_info = ""

        print(f"Epoch {epoch:3d}/{config.n_epochs} | "
              f"Train Loss: {train_loss_meter.avg:.4f} Acc: {train_acc_meter.avg:.3f} | "
              f"Val Loss: {val_loss_meter.avg:.4f} Acc: {val_acc_meter.avg:.3f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
              f"Time: {epoch_time:.1f}s{mem_info}")

        # Save best model
        if val_acc_meter.avg > best_val_acc:
            best_val_acc = val_acc_meter.avg
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': asdict(config)
            }, best_ckpt_path)
            print(f"  → Saved best model (val_acc: {best_val_acc:.4f})")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    if torch.cuda.is_available():
        print(f"Final GPU Memory: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
    print(f"{'='*60}\n")

    return history, best_ckpt_path


# ================================================================================
# STATISTICAL SIGNIFICANCE TESTING
# ================================================================================

def run_multiple_experiments(n_runs: int = 3, config: Config = CFG) -> Dict:
    """
    Run multiple independent training runs for statistical significance.

    Returns:
        Dictionary with results and statistical tests
    """
    print("\n" + "="*60)
    print(f"RUNNING {n_runs} INDEPENDENT EXPERIMENTS")
    print("="*60)

    results = []

    for run in range(1, n_runs + 1):
        print(f"\n--- Run {run}/{n_runs} ---")

        # New seed for each run
        run_seed = config.seed + run * 100
        set_seed(run_seed)

        # New data splits
        df_train, df_val, df_test = make_stratified_splits(config)

        # New model
        encoder = ConvEncoder(out_dim=config.embedding_dim, dropout=config.dropout)
        model = PrototypicalNetwork(encoder).to(DEVICE)

        # Train
        history, ckpt_path = train_model(model, df_train, df_val, config)

        # Load best and evaluate
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        metrics = evaluate_model(model, df_test, config, split='test')
        metrics['run'] = run
        metrics['seed'] = run_seed
        results.append(metrics)

        print(f"Run {run} - Test Accuracy: {metrics['accuracy']:.4f}, "
              f"F1-Macro: {metrics['f1_macro']:.4f}")

    # Statistical tests
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['f1_macro'] for r in results]

    stats_results = {
        'runs': results,
        'accuracy': {
            'mean': float(np.mean(accuracies)),
            'std': float(np.std(accuracies)),
            'min': float(np.min(accuracies)),
            'max': float(np.max(accuracies))
        },
        'f1_macro': {
            'mean': float(np.mean(f1_scores)),
            'std': float(np.std(f1_scores)),
            'min': float(np.min(f1_scores)),
            'max': float(np.max(f1_scores))
        }
    }

    print(f"\n{'='*60}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*60}")
    print(f"Accuracy:  {stats_results['accuracy']['mean']:.4f} "
          f"± {stats_results['accuracy']['std']:.4f}")
    print(f"F1-Macro:  {stats_results['f1_macro']['mean']:.4f} "
          f"± {stats_results['f1_macro']['std']:.4f}")
    print(f"{'='*60}\n")

    # Save results
    with open(os.path.join(config.output_dir, 'multi_run_stats.json'), 'w') as f:
        json.dump(stats_results, f, indent=2)

    return stats_results


# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main(config: Config = CFG, run_experiments: bool = False, n_experiment_runs: int = 3):
    """
    Main pipeline for Prototypical Networks + XAI.

    Args:
        config: Configuration object
        run_experiments: If True, run multiple experiments for statistical testing
        n_experiment_runs: Number of independent runs
    """
    print("\n" + "="*70)
    print("PROTOTYPICAL NETWORKS WITH EXPLAINABLE AI (XAI)")
    print("Research-Grade Implementation for Few-Shot Learning")
    print("="*70)

    # Setup
    set_seed(config.seed)
    gpu_available = setup_gpu()

    if run_experiments:
        # Run multiple experiments for statistical significance
        stats_results = run_multiple_experiments(n_experiment_runs, config)
        return stats_results

    # Standard single-run training
    print("\n[1/5] Creating stratified data splits...")
    df_train, df_val, df_test = make_stratified_splits(config)

    print("\n[2/5] Training Prototypical Network...")
    encoder = ConvEncoder(out_dim=config.embedding_dim, dropout=config.dropout)
    model = PrototypicalNetwork(encoder).to(DEVICE)

    # Print detailed model information
    print_model_details(model, config)

    history, best_ckpt = train_model(model, df_train, df_val, config)

    # Save training history
    with open(os.path.join(config.output_dir, 'train_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("\n[3/5] Evaluating on test set...")
    checkpoint = torch.load(best_ckpt)
    model.load_state_dict(checkpoint['model_state_dict'])

    metrics = evaluate_model(model, df_test, config, split='test')

    # Save metrics
    with open(os.path.join(config.output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print("TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"F1-Score (Macro):   {metrics['f1_macro']:.4f}")
    print(f"F1-Score (Micro):   {metrics['f1_micro']:.4f}")
    print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"ECE:                {metrics['ece']:.4f}")
    print(f"Brier Score:        {metrics['brier_score']:.4f}")
    print(f"{'='*60}\n")

    print("\n[4/5] Generating visualizations...")

    # Training history
    plot_training_history(
        history,
        os.path.join(config.plot_dir, 'training_history.png')
    )

    # Confusion matrix
    class_names = sorted(df_train['class_name'].unique())
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        os.path.join(config.plot_dir, 'confusion_matrix.png')
    )

    # Per-class metrics
    plot_per_class_metrics(
        metrics,
        class_names,
        os.path.join(config.plot_dir, 'per_class_metrics.png')
    )

    # Calibration curve (need to recompute with saved predictions)
    print("\n[5/5] Generating XAI visualizations...")
    xai_metrics = generate_xai_visualizations(
        model, df_test, config, n_samples=config.xai_samples
    )

    # Save XAI sparsity scores
    with open(os.path.join(config.output_dir, 'xai_sparsity.json'), 'w') as f:
        json.dump(xai_metrics, f, indent=2)

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': asdict(config),
        'metrics': metrics
    }, os.path.join(config.output_dir, 'final_model.pth'))

    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("="*70)
    print(f"Output directory: {config.output_dir}")
    print("\nGenerated files:")
    print("  - splits/train.csv, val.csv, test.csv")
    print("  - train_history.json")
    print("  - test_metrics.json")
    print("  - plots/training_history.png")
    print("  - plots/confusion_matrix.png")
    print("  - plots/per_class_metrics.png")
    print("  - xai/*.png (XAI visualizations)")
    print("  - checkpoints/best_model.pth")
    print("  - final_model.pth")
    print("="*70 + "\n")

    return {
        'history': history,
        'metrics': metrics,
        'xai_sparsity': xai_metrics
    }


# ================================================================================
# KAGGLE NOTEBOOK ENTRY POINT
# ================================================================================

if __name__ == '__main__':
    """
    Entry point for Kaggle Notebook execution.

    Usage in Kaggle:
    1. Upload dataset to /kaggle/input/few-shot-data/
    2. Run: !python prototypical_xai_complete.py
    3. Outputs saved to /kaggle/working/

    For multiple runs (statistical significance):
    - Set run_experiments=True
    """
    # Single training run (recommended for quick results)
    results = main(CFG, run_experiments=False)

    # For statistical significance testing (slower, runs 3 times):
    # results = main(CFG, run_experiments=True, n_experiment_runs=3)
