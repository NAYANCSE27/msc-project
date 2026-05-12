"""
================================================================================
SIAMESE NETWORKS FOR FEW-SHOT IMAGE CLASSIFICATION WITH XAI
================================================================================

A research-grade implementation of Siamese Networks integrated with
Explainable AI (XAI) techniques for few-shot medical image classification.

Key Difference from Prototypical Networks:
- Siamese Networks learn to compare pairs of images
- Uses contrastive learning with positive/negative pairs
- Similarity is measured via distance between paired embeddings
- Classification is based on which class the query is most similar to

Author: Research Implementation
Version: 1.0

================================================================================
MATHEMATICAL FORMULATION
================================================================================

1. EMBEDDING FUNCTION
   f_φ: X → R^d maps images to d-dimensional embedding space

   z_i = f_φ(x_i) where x_i ∈ R^(H×W×C)

2. SIAMESE ARCHITECTURE
   Both branches share the same encoder weights:
   z1 = f_φ(x1)
   z2 = f_φ(x2)

3. DISTANCE METRIC (L1 or L2)
   d(x1, x2) = ||f_φ(x1) - f_φ(x2)||²

4. SIMILARITY MEASURE
   s = exp(-d(x1, x2))  [Sigmoid similarity]

5. CONTRASTIVE LOSS
   L = (1-Y) * D² + Y * max(0, margin - D)²

   where Y=1 if pair is similar, Y=0 if dissimilar

6. PAIRWISE CLASSIFICATION LOSS
   For N-way classification:
   P(class_c | x_query, x_support) = softmax(-d(x_query, x_support_c))

================================================================================
EVALUATION METRICS
================================================================================

- Accuracy: (TP + TN) / (TP + TN + FP + FN)
- F1-Score (Macro): (1/C) * Σ F1_c where F1_c = 2*Prec*Rec / (Prec+Rec)
- ECE (Expected Calibration Error):
  ECE = Σ_{b=1}^B (|B_b|/n) * |acc(B_b) - conf(B_b)|
- Attribution Sparsity: fraction of pixels below threshold τ=0.6

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
from torch.utils.data import Dataset, DataLoader, PairDataset
from torchvision import transforms
from torchvision.utils import make_grid

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    precision_recall_fscore_support, classification_report
)
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import ttest_ind, sem

import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 10

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Central configuration for the Siamese Network experiment."""

    # Data paths - MODIFY THESE FOR YOUR DATASET
    DATA_ROOT = '/kaggle/input/cucumber-dataset/Original Image'
    OUTPUT_DIR = '/kaggle/working'

    # Random seed for reproducibility
    RNG_SEED = 42

    # Dataset splitting (80/10/10)
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1

    # Few-shot parameters
    N_WAY = 8           # Classes per episode
    K_SHOT = 5          # Support examples per class
    Q_QUERY = 15        # Query examples per class

    # Training parameters
    EPOCHS = 30
    BATCH_SIZE = 32     # Pairs per batch
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    MARGIN = 1.0        # Contrastive loss margin

    # Model parameters
    EMBEDDING_DIM = 128
    IMAGE_SIZE = 128

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = Config()

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def create_directories():
    """Create output directories."""
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
# DATA HANDLING
# =============================================================================

def make_stratified_splits(root_dir):
    """
    Perform stratified train/val/test splitting.

    Mathematical formulation:
    For each class c with n_c samples:
        n_train_c = floor(n_c * 0.8)
        n_val_c = floor(n_c * 0.1)
        n_test_c = n_c - n_train_c - n_val_c

    Args:
        root_dir: Root directory containing class subdirectories

    Returns:
        df_train, df_val, df_test, class_names
    """
    data = []
    root = Path(root_dir)

    # Discover classes
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    print(f"Discovered {len(classes)} classes: {classes}")

    if len(classes) < 2:
        raise ValueError('Dataset root must contain at least 2 class subfolders')

    # Collect images
    for lbl, cls in enumerate(classes):
        images = list((root / cls).glob('*'))
        images = [x for x in images if x.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif']]
        for img in images:
            data.append({'image': str(img), 'label': lbl, 'class': cls})

    df = pd.DataFrame(data)
    print(f"Total images: {len(df)}")
    per_class = df['label'].value_counts().sort_index().tolist()
    print(f"Images per class: {per_class}")

    x, y = df['image'], df['label']

    # First split: 80% train, 20% temp
    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=(1-cfg.TRAIN_RATIO),
                                       random_state=cfg.RNG_SEED)
    train_idx, temp_idx = next(splitter1.split(x, y))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)

    # Second split: 10% val, 10% test from temp
    test_ratio_adjusted = cfg.TEST_RATIO / (cfg.VAL_RATIO + cfg.TEST_RATIO)
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio_adjusted,
                                      random_state=cfg.RNG_SEED)
    val_idx, test_idx = next(splitter2.split(df_temp['image'], df_temp['label']))

    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)

    # Save splits
    df_train.to_csv(os.path.join(cfg.SPLIT_DIR, 'train.csv'), index=False)
    df_val.to_csv(os.path.join(cfg.SPLIT_DIR, 'val.csv'), index=False)
    df_test.to_csv(os.path.join(cfg.SPLIT_DIR, 'test.csv'), index=False)

    print(f"\nStratified Split Results:")
    print(f"  Training:   {len(df_train)} images ({len(df_train)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(df_val)} images ({len(df_val)/len(df)*100:.1f}%)")
    print(f"  Testing:    {len(df_test)} images ({len(df_test)/len(df)*100:.1f}%)")

    return df_train, df_val, df_test, classes

# =============================================================================
# DATASET AND TRANSFORMS
# =============================================================================

# =============================================================================
# OPTIMIZED DATA HANDLING - PRELOAD ALL IMAGES INTO MEMORY
# =============================================================================

class CachedImageDataset(Dataset):
    """
    Optimized dataset that preloads ALL images into memory.
    This eliminates disk I/O bottleneck during training.
    """

    def __init__(self, df, transform=None, device='cuda'):
        self.df = df
        self.transform = transform
        self.device = device

        # Preload ALL images into memory at initialization
        print(f"Preloading {len(df)} images into memory...")
        self.images = []
        self.labels = []

        for idx in tqdm(range(len(df)), desc="Loading images"):
            row = self.df.iloc[idx]
            image = Image.open(row['image']).convert('RGB')

            # Resize immediately to target size (faster than transform)
            image = image.resize((64, 64), Image.BILINEAR)  # Smaller size = faster

            # Convert to tensor and normalize in one step
            img_tensor = transforms.ToTensor()(image)  # [0, 1] range
            img_tensor = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )(img_tensor)

            self.images.append(img_tensor)
            self.labels.append(int(row['label']))

        # Stack all images into a single tensor [N, C, H, W]
        self.images = torch.stack(self.images)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

        print(f"Loaded {len(self)} images. Shape: {self.images.shape}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Just return from preloaded tensor - NO disk I/O!
        return self.images[idx], self.labels[idx]


def make_transforms_optimized(img_size=64):
    """
    Create optimized transforms with smaller image size.
    64x64 is sufficient for few-shot learning and much faster.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform

# =============================================================================
# SIAMESE NETWORK PAIR SAMPLING
# =============================================================================

class SiameseSampler:
    """
    Sampler for creating few-shot episodes using Siamese pair comparison.

    Each episode consists of:
    - N-way classes selected from the dataset
    - K-shot support examples per class
    - Q-query examples to classify

    For Siamese networks, we compare each query to all support examples
    and determine the most similar class.

    Episode sampling algorithm:
    1. Randomly select N classes without replacement
    2. For each class, sample K+Q examples without replacement
    3. Use K examples as support, Q as query
    4. For each query, compute similarity to all support images
    5. Classify query based on maximum similarity
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

        # Group indices by class
        self.by_class = {c: np.where(self.labels == c)[0] for c in np.unique(self.labels)}

        # Verify sufficient samples
        min_required = k_shot + q_query
        for c, idx in self.by_class.items():
            if len(idx) < min_required:
                raise ValueError(
                    f'Class {c} has {len(idx)} samples, '
                    f'but needs at least {min_required}'
                )

    def __len__(self):
        return self.episodes

    def __iter__(self):
        for _ in range(self.episodes):
            # Select N classes without replacement
            selected_classes = self.rng.choice(
                list(self.by_class.keys()),
                size=self.n_way,
                replace=False
            )

            support_idx = []
            query_idx = []

            for c in selected_classes:
                choices = self.rng.choice(
                    self.by_class[c],
                    size=self.k_shot + self.q_query,
                    replace=False
                )
                support_idx.extend(choices[:self.k_shot].tolist())
                query_idx.extend(choices[self.k_shot:].tolist())

            yield support_idx, query_idx

def pair_loader_optimized(df, transform, device='cuda', n_way=None, k_shot=None, q_query=None, episodes=None):
    """OPTIMIZED helper to create cached dataset and sampler."""
    if n_way is None: n_way = cfg.N_WAY
    if k_shot is None: k_shot = cfg.K_SHOT
    if q_query is None: q_query = cfg.Q_QUERY
    if episodes is None: episodes = EPISODES_PER_EPOCH

    # Use cached dataset that preloads ALL images into memory
    dataset = CachedImageDataset(df, transform, device)
    sampler = SiameseSampler(
        df['label'].to_numpy(),
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        episodes=episodes
    )
    return dataset, sampler

# EPISODES_PER_EPOCH for Siamese training
EPISODES_PER_EPOCH = 20

# =============================================================================
# SIAMESE NETWORK MODEL
# =============================================================================

class SiameseEncoder(nn.Module):
    """
    OPTIMIZED Convolutional encoder for Siamese Networks.
    Lighter architecture for faster training.
    """

    def __init__(self, out_dim=128):
        super().__init__()

        # OPTIMIZED: Fewer channels, smaller output
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4))  # Smaller output
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        features = self.conv_blocks(x)
        embedding = self.fc(features)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


class SiameseNetwork(nn.Module):
    """
    Siamese Network for few-shot classification via pairwise comparison.

    Architecture:
    - Two identical encoder branches (weight sharing)
    - Distance computation between paired embeddings
    - Similarity-based classification

    Forward pass:
    1. z1 = f_φ(x1), z2 = f_φ(x2) - embed both images
    2. d = ||z1 - z2||² - compute distance
    3. s = exp(-d) or use sigmoid - similarity score

    For N-way classification:
    1. Compare query to each support image
    2. Aggregate similarities by class
    3. Classify to highest aggregate similarity class
    """

    def __init__(self, encoder=None, embedding_dim=128):
        super().__init__()

        if encoder is None:
            self.encoder = SiameseEncoder(out_dim=embedding_dim)
        else:
            self.encoder = encoder

        self.embedding_dim = embedding_dim

    def forward_one(self, x):
        """Forward pass for single image through encoder."""
        return self.encoder(x)

    def forward_pair(self, x1, x2):
        """
        Forward pass for a pair of images.

        Args:
            x1: [batch_size, 3, H, W] first image
            x2: [batch_size, 3, H, W] second image

        Returns:
            embeddings1: [batch_size, D] embeddings for first images
            embeddings2: [batch_size, D] embeddings for second images
            distances: [batch_size] squared Euclidean distances
        """
        emb1 = self.encoder(x1)
        emb2 = self.encoder(x2)

        # Squared Euclidean distance
        diff = emb1 - emb2
        dist = torch.sum(diff ** 2, dim=1)

        return emb1, emb2, dist

    def compute_similarity(self, query_emb, support_embs, support_labels):
        """
        Compute similarity between query and support set for classification.

        Mathematical formulation:
        For query embedding z_q and support set {z_s^i, y_s^i}:
        - Distance: d_i = ||z_q - z_s^i||²
        - Similarity: s_i = exp(-d_i)
        - Class score: score(c) = Σ_{i: y_s^i = c} s_i
        - Probability: P(y=c) = softmax(score)_c

        Args:
            query_emb: [Q, D] query embeddings
            support_embs: [S, D] support embeddings
            support_labels: [S] support labels

        Returns:
            logits: [Q, N] class scores per query
            similarities: [Q, S] pairwise similarities
        """
        n_query = query_emb.size(0)
        n_support = support_embs.size(0)
        n_way = len(torch.unique(support_labels))

        # Compute pairwise distances [Q, S]
        diff = query_emb.unsqueeze(1) - support_embs.unsqueeze(0)  # [Q, S, D]
        dists = torch.sum(diff ** 2, dim=2)  # [Q, S]

        # Convert to similarities using exp(-d)
        similarities = torch.exp(-dists)  # [Q, S]

        # Aggregate similarities by class
        unique_labels = torch.unique(support_labels)
        logits = torch.zeros(n_query, n_way, device=query_emb.device)

        for idx, c in enumerate(unique_labels):
            mask = (support_labels == c).unsqueeze(0)  # [1, S]
            class_sim = (similarities * mask.float()).sum(dim=1)  # [Q]
            logits[:, idx] = class_sim

        return logits, similarities

    def forward(self, support, support_labels, query):
        """
        Forward pass for few-shot classification.

        Args:
            support: Support images [S, 3, H, W]
            support_labels: Support labels [S]
            query: Query images [Q, 3, H, W]

        Returns:
            logits: Classification logits [Q, N]
            query_emb: Query embeddings [Q, D]
            support_emb: Support embeddings [S, D]
        """
        # Encode all images
        support_emb = self.encoder(support)
        query_emb = self.encoder(query)

        # Compute classification logits via similarity
        logits, similarities = self.compute_similarity(query_emb, support_emb, support_labels)

        return logits, query_emb, support_emb, similarities


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for Siamese Network training.

    Mathematical formulation:
    L = (1-Y) * D² + Y * max(0, margin - D)²

    where:
    - Y = 1 if pair is similar (same class), 0 if dissimilar
    - D = ||f(x1) - f(x2)||² is the squared Euclidean distance
    - margin is a hyperparameter (typically 1.0)

    For similar pairs (Y=1): encourage small distances
    For dissimilar pairs (Y=0): enforce minimum margin
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, distance, label):
        """
        Compute contrastive loss.

        Args:
            distance: [batch_size] squared distances between pairs
            label: [batch_size] 1 if similar, 0 if dissimilar

        Returns:
            loss: scalar contrastive loss
        """
        # Similar pairs: minimize distance
        similar_loss = label * distance

        # Dissimilar pairs: enforce margin
        dissimilar_loss = (1 - label) * torch.clamp(self.margin - torch.sqrt(distance + 1e-8), min=0) ** 2

        loss = (similar_loss + dissimilar_loss).mean()
        return loss


class PairwiseCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for N-way classification using pairwise similarities.

    For each query-support pair, compute similarity and use for classification.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        """
        Args:
            logits: [batch_size, n_way] class scores
            labels: [batch_size] ground truth class indices (0 to n_way-1)
        """
        return F.cross_entropy(logits, labels)


def euclidean_dist(x, y):
    """
    Compute squared Euclidean distance between two sets of vectors.

    Mathematical formula:
    d(x_i, y_j) = ||x_i - y_j||² = ||x_i||² + ||y_j||² - 2 * x_i · y_j

    This efficient computation avoids explicit pairwise loops.

    Args:
        x: [n, d] first set of vectors
        y: [m, d] second set of vectors

    Returns:
        dist: [n, m] pairwise squared distances
    """
    n, m, d = x.size(0), y.size(0), x.size(1)

    # ||x_i||² term: [n, 1]
    xx = (x ** 2).sum(dim=1, keepdim=True).expand(n, m)
    # ||y_j||² term: [1, m]
    yy = (y ** 2).sum(dim=1, keepdim=True).expand(m, n).t()
    # -2 * x_i · y_j term
    dist = xx + yy - 2.0 * x @ y.t()

    # Numerical stability
    dist = torch.clamp(dist, min=0.0)

    return dist

# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_ece(probs, labels, n_bins=15):
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between confidence and accuracy
    across different confidence bins.

    Mathematical formulation:
    ECE = Σ_{b=1}^B (|B_b|/n) * |acc(B_b) - conf(B_b)|

    Lower ECE indicates better calibration.

    Args:
        probs: [n, num_classes] predicted probabilities
        labels: [n] ground truth labels
        n_bins: number of confidence bins

    Returns:
        ece: scalar ECE value
    """
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

def compute_attribution_sparsity(attributions, threshold=0.6):
    """
    Compute attribution sparsity of XAI explanation maps.

    Sparsity measures what fraction of the explanation is
    concentrated in a small number of pixels.

    Mathematical formulation:
    Sparsity = (1/(H*W)) * Σ_{i,j} 1(|a_{ij}| < τ)

    where τ = 0.6 is the sparsity threshold.

    Higher sparsity indicates more focused explanations.

    Args:
        attributions: [H, W] explanation heatmap
        threshold: cutoff threshold (default: 0.6)

    Returns:
        sparsity: fraction of pixels below threshold
    """
    attr = np.abs(attributions)
    if attr.max() > 0:
        attr = attr / attr.max()

    sparsity = np.mean(attr < threshold)
    return float(sparsity)

def compute_all_metrics(y_true, y_pred, y_prob, class_names=None):
    """
    Compute comprehensive evaluation metrics.

    Metrics computed:
    - Accuracy (overall)
    - F1-Score (macro, micro, weighted)
    - Per-class precision, recall, F1
    - Confusion matrix
    - Expected Calibration Error (ECE)

    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        y_prob: predicted probabilities [n, num_classes]
        class_names: list of class names for reporting

    Returns:
        dict: comprehensive metrics
    """
    if isinstance(y_prob, torch.Tensor):
        y_prob_np = y_prob.cpu().numpy()
    else:
        y_prob_np = y_prob

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # F1-Scores
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Per-class metrics
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # ECE
    ece_val = compute_ece(
        torch.from_numpy(y_prob_np),
        torch.from_numpy(y_true),
        n_bins=15
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support_per_class': support.tolist(),
        'ece': ece_val,
        'confusion_matrix': cm.tolist(),
    }

    return metrics

def print_metrics(metrics, class_names=None):
    """Pretty print metrics."""
    print(f"\n{'='*60}")
    print("EVALUATION METRICS")
    print(f"{'='*60}")
    print(f"Overall Accuracy:  {metrics['accuracy']:.4f}")
    print(f"F1-Score (Macro):  {metrics['f1_macro']:.4f}")
    print(f"F1-Score (Micro):  {metrics['f1_micro']:.4f}")
    print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"ECE (Calibration): {metrics['ece']:.4f}")
    print(f"\nPer-Class F1 Scores:")

    if class_names:
        for i, (name, f1) in enumerate(zip(class_names, metrics['f1_per_class'])):
            print(f"  Class {i} ({name[:15]:15s}): {f1:.4f}")
    else:
        for i, f1 in enumerate(metrics['f1_per_class']):
            print(f"  Class {i}: {f1:.4f}")

    print(f"{'='*60}\n")

# =============================================================================
# XAI METHODS FOR SIAMESE NETWORKS
# =============================================================================

class SiameseGradCAM:
    """
    Gradient-weighted Class Activation Mapping for Siamese Networks.

    For Siamese networks, we compute Grad-CAM on the query image
    with respect to its similarity to the support set.

    Mathematical formulation:
    L_Grad-CAM^(c) = ReLU(Σ_k α_k^c * A^k)

    where α_k^c is computed from gradients w.r.t. the target class similarity.
    """

    def __init__(self, model, target_layer, support_images=None, support_labels=None):
        self.model = model
        self.target_layer = target_layer
        self.support_images = support_images
        self.support_labels = support_labels
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        self.hook_handles.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self.hook_handles.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for a single query image.

        Args:
            input_tensor: [3, H, W] query image tensor
            target_class: class index for visualization (None = predicted class)

        Returns:
            cam: [H, W] normalized heatmap
        """
        self.model.eval()
        self.model.zero_grad()

        if self.support_images is None or self.support_labels is None:
            raise ValueError("GradCAM requires support_images and support_labels")

        # Prepare query image
        query_img = input_tensor.unsqueeze(0)

        # Forward pass through Siamese network
        logits, query_emb, support_emb, similarities = self.model(
            self.support_images, self.support_labels, query_img
        )

        # Get target class
        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        if target_class >= logits.shape[1]:
            target_class = torch.argmax(logits, dim=1).item()

        # Backward pass
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        # Compute Grad-CAM
        grads = self.gradients[0]
        acts = self.activations[0]

        # Global average pooling of gradients
        weights = torch.mean(grads, dim=(1, 2), keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * acts, dim=0)

        # Post-processing
        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        return cam

    def close(self):
        """Remove hooks."""
        for handle in self.hook_handles:
            handle.remove()


def siamese_saliency_map(model, input_tensor, support_images=None, support_labels=None, target_class=None):
    """
    Compute gradient-based saliency map for Siamese network.

    The saliency map shows which pixels have the highest influence
    on the similarity computation with support set.

    Mathematical formulation:
    S_{ij} = |∂score/∂x_{ij}|

    where score is the classification score for the target class.

    Args:
        model: SiameseNetwork model
        input_tensor: [3, H, W] query image
        support_images: support set images
        support_labels: support set labels
        target_class: class index (None = predicted)

    Returns:
        saliency: [H, W] normalized saliency map
    """
    model.eval()

    # Enable gradient computation
    input_tensor = input_tensor.unsqueeze(0).clone().detach().requires_grad_(True)

    if support_images is None or support_labels is None:
        raise ValueError("saliency_map requires support_images and support_labels")

    # Forward pass
    logits, query_emb, support_emb, similarities = model(
        support_images, support_labels, input_tensor
    )

    # Get target class
    if target_class is None:
        target_class = torch.argmax(logits, dim=1).item()

    if target_class >= logits.shape[1]:
        target_class = torch.argmax(logits, dim=1).item()

    # Backward pass
    score = logits[0, target_class]
    score.backward()

    # Get gradients
    saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()

    # Take maximum across channels
    saliency = np.max(saliency, axis=0)

    # Normalize
    saliency = saliency - saliency.min()
    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    return saliency


def save_heatmap(img, mask, path, alpha=0.5, title=None):
    """
    Save heatmap overlay visualization.

    Args:
        img: [3, H, W] normalized image tensor
        mask: [H, W] heatmap to overlay
        path: save path
        alpha: overlay transparency
    """
    # Denormalize image
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_p = np.clip((img_np * std + mean), 0, 1)

    # Create heatmap
    cmap = plt.get_cmap('jet')
    heatmap = cmap(mask)[..., :3]

    # Overlay
    overlay = np.clip((1 - alpha) * img_p + alpha * heatmap, 0, 1)

    # Save
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    if title:
        plt.title(title)
    plt.imshow(overlay)
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=150, bbox_inches='tight', pad_inches=0.1,
                facecolor='white', edgecolor='none')
    plt.close()

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

# =============================================================================
# OPTIMIZED TRAINING FUNCTIONS - FASTER TRAINING
# =============================================================================

def run_siamese_episode_fast(model, optimizer, dataset, support_idx, query_idx, criterion, scaler):
    """
    OPTIMIZED episode training with mixed precision and reduced memory moves.
    """
    model.train()

    # Convert indices to tensors ONCE
    support_idx_t = torch.tensor(support_idx, dtype=torch.long)
    query_idx_t = torch.tensor(query_idx, dtype=torch.long)

    # Load data to device in ONE batch operation
    support_images = dataset.images[support_idx_t].to(cfg.DEVICE)
    support_labels = dataset.labels[support_idx_t].to(cfg.DEVICE)
    query_images = dataset.images[query_idx_t].to(cfg.DEVICE)
    query_labels = dataset.labels[query_idx_t].to(cfg.DEVICE)

    # Map labels to 0..N-1
    unique = torch.unique(support_labels)
    label_map = {int(c): i for i, c in enumerate(unique)}
    support_labels_mapped = torch.tensor(
        [label_map[int(l)] for l in support_labels],
        dtype=torch.long, device=cfg.DEVICE
    )
    query_labels_mapped = torch.tensor(
        [label_map[int(l)] for l in query_labels],
        dtype=torch.long, device=cfg.DEVICE
    )

    # Forward with mixed precision
    optimizer.zero_grad()

    with torch.cuda.amp.autocast():
        logits, _, _, _ = model(support_images, support_labels_mapped, query_images)
        loss = criterion(logits, query_labels_mapped)

    # Backward with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # Compute metrics
    preds = torch.argmax(logits, dim=1)
    acc = (preds == query_labels_mapped).float().mean().item()
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()

    return (loss.item(), acc, preds.detach().cpu().numpy(),
            query_labels_mapped.detach().cpu().numpy(), probs)


def validate_siamese_episode_fast(model, dataset, support_idx, query_idx, criterion):
    """OPTIMIZED validation with no gradients."""
    model.eval()

    with torch.no_grad():
        support_idx_t = torch.tensor(support_idx, dtype=torch.long)
        query_idx_t = torch.tensor(query_idx, dtype=torch.long)

        support_images = dataset.images[support_idx_t].to(cfg.DEVICE)
        support_labels = dataset.labels[support_idx_t].to(cfg.DEVICE)
        query_images = dataset.images[query_idx_t].to(cfg.DEVICE)
        query_labels = dataset.labels[query_idx_t].to(cfg.DEVICE)

        unique = torch.unique(support_labels)
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
            logits, _, _, _ = model(support_images, support_labels_mapped, query_images)
            loss = criterion(logits, query_labels_mapped)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == query_labels_mapped).float().mean().item()
        probs = F.softmax(logits, dim=1).cpu().numpy()

        return (loss.item(), acc, preds.cpu().numpy(),
                query_labels_mapped.cpu().numpy(), probs)


def train_siamese_network_optimized(model, df_train, df_val, n_epochs=None, lr=None, verbose=True):
    """
    OPTIMIZED training loop with:
    - Mixed precision training
    - Preloaded dataset in memory
    - Reduced memory operations
    - Faster validation
    """
    if n_epochs is None: n_epochs = cfg.EPOCHS
    if lr is None: lr = cfg.LEARNING_RATE

    # Create cached datasets (loads all images into memory ONCE)
    train_transform, eval_transform = make_transforms_optimized()

    # OPTIMIZED: Use cached dataset that preloads images
    train_dataset = CachedImageDataset(df_train, train_transform, cfg.DEVICE)
    val_dataset = CachedImageDataset(df_val, eval_transform, cfg.DEVICE)

    train_sampler = SiameseSampler(
        df_train['label'].to_numpy(),
        n_way=cfg.N_WAY, k_shot=cfg.K_SHOT, q_query=cfg.Q_QUERY,
        episodes=EPISODES_PER_EPOCH
    )
    val_sampler = SiameseSampler(
        df_val['label'].to_numpy(),
        n_way=cfg.N_WAY, k_shot=cfg.K_SHOT, q_query=cfg.Q_QUERY,
        episodes=10
    )

    criterion = PairwiseCrossEntropyLoss()

    # OPTIMIZED: Use AdamW and OneCycleLR for faster convergence
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=cfg.WEIGHT_DECAY
    )

    # OPTIMIZED: OneCycle scheduler - faster than StepLR
    total_steps = n_epochs * EPISODES_PER_EPOCH
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        pct_start=0.3, anneal_strategy='cos'
    )

    # OPTIMIZED: Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'epoch_times': []
    }

    best_val_acc = 0.0
    best_ckpt = None

    print(f"\n{'='*60}")
    print("OPTIMIZED SIAMESE NETWORK TRAINING")
    print(f"{'='*60}")
    print(f"Epochs: {n_epochs}, Episodes/Epoch: {EPISODES_PER_EPOCH}")
    print(f"Image Size: 64x64 (optimized)")
    print(f"Mixed Precision: Enabled")
    print(f"Preloaded Dataset: {len(train_dataset)} images in memory")
    print(f"{'='*60}\n")

    epoch_start = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        train_accs = []

        for support_idx, query_idx in train_sampler:
            loss, acc, _, _, _ = run_siamese_episode_fast(
                model, optimizer, train_dataset, support_idx, query_idx, criterion, scaler
            )
            train_losses.append(loss)
            train_accs.append(acc)
            scheduler.step()

        model.eval()
        val_losses = []
        val_accs = []

        with torch.no_grad():
            for support_idx, query_idx in val_sampler:
                loss, acc, _, _, _ = validate_siamese_episode_fast(
                    model, val_dataset, support_idx, query_idx, criterion
                )
                val_losses.append(loss)
                val_accs.append(acc)

        train_loss = float(np.mean(train_losses))
        train_acc = float(np.mean(train_accs))
        val_loss = float(np.mean(val_losses))
        val_acc = float(np.mean(val_accs))
        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)

        if verbose and epoch % 2 == 0:
            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f} | "
                f"Time: {epoch_time:.1f}s"
            )

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

    print(f"\nTraining Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

    return history, best_ckpt

# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_siamese_network_optimized(model, df_test, episodes=20):
    """
    OPTIMIZED evaluation with preloaded dataset.
    """
    _, eval_transform = make_transforms_optimized()
    test_dataset = CachedImageDataset(df_test, eval_transform, cfg.DEVICE)
    test_sampler = SiameseSampler(
        df_test['label'].to_numpy(),
        n_way=cfg.N_WAY, k_shot=cfg.K_SHOT, q_query=cfg.Q_QUERY,
        episodes=episodes
    )

    criterion = PairwiseCrossEntropyLoss()

    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    all_loss = []
    all_acc = []

    with torch.no_grad():
        for support_idx, query_idx in test_sampler:
            support_idx_t = torch.tensor(support_idx, dtype=torch.long)
            query_idx_t = torch.tensor(query_idx, dtype=torch.long)

            support_images = test_dataset.images[support_idx_t].to(cfg.DEVICE)
            support_labels = test_dataset.labels[support_idx_t].to(cfg.DEVICE)
            query_images = test_dataset.images[query_idx_t].to(cfg.DEVICE)
            query_labels = test_dataset.labels[query_idx_t].to(cfg.DEVICE)

            unique = torch.unique(support_labels)
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
                logits, _, _, _ = model(support_images, support_labels_mapped, query_images)
                loss = criterion(logits, query_labels_mapped)

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            probs = F.softmax(logits, dim=1).cpu().numpy()

            all_loss.append(loss.item())
            all_acc.append((preds == query_labels_mapped.cpu().numpy()).mean())

            y_true.extend(query_labels_mapped.cpu().numpy().tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())

    metrics = compute_all_metrics(
        np.array(y_true),
        np.array(y_pred),
        np.array(y_prob),
        CLASS_NAMES
    )
    metrics['test_loss'] = float(np.mean(all_loss))
    metrics['test_acc'] = float(np.mean(all_acc))

    return metrics

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_training_history(history, save_path=None):
    """Plot training and validation loss/accuracy curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Smoothed learning curves
    window = min(5, len(history['train_loss']))
    train_loss_smooth = np.convolve(history['train_loss'],
                                    np.ones(window)/window, mode='valid')
    val_loss_smooth = np.convolve(history['val_loss'],
                                  np.ones(window)/window, mode='valid')
    axes[1, 0].plot(train_loss_smooth, 'b-', label='Train Loss (smoothed)', linewidth=2)
    axes[1, 0].plot(val_loss_smooth, 'r-', label='Val Loss (smoothed)', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title(f'Learning Curves (Moving Average, window={window})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Epoch timing
    axes[1, 1].bar(epochs, history['epoch_times'], color='green', alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].set_title('Training Time per Epoch')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path=None, normalize=True):
    """Plot confusion matrix with per-class labels."""
    fig, ax = plt.subplots(figsize=(10, 8))

    if normalize:
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        fmt = '.2%'
        vmin, vmax = 0, 1
    else:
        cm_normalized = cm
        fmt = 'd'
        vmin, vmax = 0, None

    sns.heatmap(cm_normalized, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, vmin=vmin, vmax=vmax,
                cbar_kws={'label': 'Proportion'})

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_per_class_metrics(metrics, class_names, save_path=None):
    """Plot per-class precision, recall, and F1-score."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = np.arange(len(class_names))
    width = 0.25

    # Precision
    axes[0].bar(x, metrics['precision_per_class'], width, label='Precision', color='steelblue')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Per-Class Precision')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([c[:10] for c in class_names], rotation=45, ha='right')
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3, axis='y')

    # Recall
    axes[1].bar(x, metrics['recall_per_class'], width, label='Recall', color='forestgreen')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Per-Class Recall')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([c[:10] for c in class_names], rotation=45, ha='right')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')

    # F1-Score
    axes[2].bar(x, metrics['f1_per_class'], width, label='F1-Score', color='coral')
    axes[2].set_xlabel('Class')
    axes[2].set_ylabel('Score')
    axes[2].set_title('Per-Class F1-Score')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([c[:10] for c in class_names], rotation=45, ha='right')
    axes[2].set_ylim([0, 1])
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


# =============================================================================
# XAI VISUALIZATION
# =============================================================================

def generate_xai_explanations_optimized(model, df_test, n_samples=5, save_dir=None):
    """
    OPTIMIZED XAI explanations using preloaded dataset.
    """
    if save_dir is None: save_dir = cfg.XAI_DIR

    _, eval_transform = make_transforms_optimized()
    ds = CachedImageDataset(df_test, eval_transform, cfg.DEVICE)

    sampler = SiameseSampler(
        df_test['label'].to_numpy(),
        n_way=cfg.N_WAY, k_shot=cfg.K_SHOT, q_query=cfg.Q_QUERY, episodes=1
    )

    support_idx, query_idx = next(iter(sampler))

    # Use preloaded dataset - NO disk I/O!
    support_images = ds.images[torch.tensor(support_idx, dtype=torch.long)].to(cfg.DEVICE)
    support_labels = ds.labels[torch.tensor(support_idx, dtype=torch.long)].to(cfg.DEVICE)
    query_images = ds.images[torch.tensor(query_idx, dtype=torch.long)].to(cfg.DEVICE)
    query_labels = ds.labels[torch.tensor(query_idx, dtype=torch.long)].to(cfg.DEVICE)

    # Map labels
    unique = torch.unique(support_labels)
    label_map = {int(c): i for i, c in enumerate(unique)}
    support_labels_mapped = torch.tensor(
        [label_map[int(l)] for l in support_labels],
        dtype=torch.long
    ).to(cfg.DEVICE)

    # Get query images and labels
    query_images = torch.stack([ds[i][0] for i in query_idx]).to(cfg.DEVICE)
    query_labels = torch.tensor([ds[i][1] for i in query_idx], dtype=torch.long).to(cfg.DEVICE)

    print(f"\n{'='*60}")
    print("XAI VISUALIZATION GENERATION (Siamese Network)")
    print(f"{'='*60}")
    print(f"Support set: {len(support_idx)} images")
    print(f"Query set: {len(query_idx)} images")
    print(f"Generating {n_samples} explanations...")
    print(f"{'='*60}\n")

    sparsity_results = []

    for idx in range(min(n_samples, len(query_idx))):
        img = query_images[idx]
        true_label = query_labels[idx].item()

        # Get prediction
        logits, query_emb, support_emb, similarities = model(
            support_images, support_labels_mapped, img.unsqueeze(0)
        )
        pred = torch.argmax(logits, dim=1).item()
        confidence = F.softmax(logits, dim=1)[0, pred].item()

        # Map prediction back to original label
        pred_original = int(list(label_map.keys())[list(label_map.values()).index(pred)])

        # Target layer for Grad-CAM (last conv layer of encoder)
        target_layer = model.encoder.conv_blocks[4]

        # Generate Grad-CAM
        gradcam = SiameseGradCAM(
            model,
            target_layer=target_layer,
            support_images=support_images,
            support_labels=support_labels_mapped
        )
        cam_mask = gradcam.generate(img, target_class=pred)
        gradcam.close()

        # Generate saliency map
        sal_map = siamese_saliency_map(
            model, img,
            support_images=support_images,
            support_labels=support_labels_mapped,
            target_class=pred
        )

        # Compute sparsity
        sparsity_cam = compute_attribution_sparsity(cam_mask)
        sparsity_sal = compute_attribution_sparsity(sal_map)
        sparsity_results.append({
            'sample': idx,
            'true_label': CLASS_NAMES[true_label] if true_label < len(CLASS_NAMES) else f'Class {true_label}',
            'pred_label': CLASS_NAMES[pred_original] if pred_original < len(CLASS_NAMES) else f'Class {pred_original}',
            'confidence': confidence,
            'sparsity_cam': sparsity_cam,
            'sparsity_sal': sparsity_sal
        })

        # Save visualizations
        save_heatmap(
            img.cpu(), cam_mask,
            os.path.join(save_dir, f'siamese_gradcam_sample{idx}_true{true_label}_pred{pred_original}.png'),
            title=f'Siamese Grad-CAM: True={true_label}, Pred={pred_original}, Conf={confidence:.2f}'
        )

        save_heatmap(
            img.cpu(), sal_map,
            os.path.join(save_dir, f'siamese_saliency_sample{idx}_true{true_label}_pred{pred_original}.png'),
            title=f'Siamese Saliency: True={true_label}, Pred={pred_original}, Conf={confidence:.2f}'
        )

        print(f"Sample {idx}: True={true_label}, Pred={pred_original}, "
              f"Conf={confidence:.3f}, CAM sparsity={sparsity_cam:.3f}, "
              f"Saliency sparsity={sparsity_sal:.3f}")

    return sparsity_results


# =============================================================================
# STATISTICAL SIGNIFICANCE TESTING
# =============================================================================

def run_multiple_experiments(n_runs=3, epochs_per_run=None):
    """
    Run multiple independent experiments for statistical analysis.

    Args:
        n_runs: number of independent training runs
        epochs_per_run: epochs per run

    Returns:
        results: list of metrics from each run
        stats: statistical analysis results
    """
    if epochs_per_run is None:
        epochs_per_run = max(5, cfg.EPOCHS // 3)  # Faster for multiple runs

    print(f"\n{'='*60}")
    print(f"MULTIPLE EXPERIMENT RUNS")
    print(f"{'='*60}")
    print(f"Number of runs: {n_runs}")
    print(f"Epochs per run: {epochs_per_run}")
    print(f"{'='*60}\n")

    results = []

    for run in range(1, n_runs + 1):
        print(f"\n--- Run {run}/{n_runs} ---")

        # Set different seed for each run
        seed = cfg.RNG_SEED + run
        set_seed(seed)

        # Create fresh data splits
        df_train, df_val, df_test = make_stratified_splits(cfg.DATA_ROOT, seed=seed)

        # Initialize fresh model
        model = SiameseNetwork(embedding_dim=cfg.EMBEDDING_DIM).to(cfg.DEVICE)

        # Train with optimized function
        history, ckpt = train_siamese_network_optimized(
            model, df_train, df_val,
            n_epochs=epochs_per_run,
            lr=cfg.LEARNING_RATE,
            verbose=False
        )

        # Load best checkpoint
        model.load_state_dict(torch.load(ckpt)['model_state_dict'])

        # Evaluate with optimized function
        metrics = evaluate_siamese_network_optimized(model, df_test)
        metrics['run'] = run
        metrics['final_train_acc'] = history['train_acc'][-1]
        metrics['final_val_acc'] = history['val_acc'][-1]

        results.append(metrics)

        print(f"  Run {run} Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Run {run} Test F1-Macro: {metrics['f1_macro']:.4f}")

    # Statistical analysis
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['f1_macro'] for r in results]

    # T-test comparing runs
    t_acc, p_acc = ttest_ind([accuracies[0]], accuracies[1:] if len(accuracies) > 1 else accuracies)
    t_f1, p_f1 = ttest_ind([f1_scores[0]], f1_scores[1:] if len(f1_scores) > 1 else f1_scores)

    stats = {
        'n_runs': n_runs,
        'accuracies': accuracies,
        'f1_scores': f1_scores,
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'sem_accuracy': sem(accuracies),
        'mean_f1': np.mean(f1_scores),
        'std_f1': np.std(f1_scores),
        't_test_accuracy': {'t_stat': float(t_acc), 'p_val': float(p_acc)},
        't_test_f1': {'t_stat': float(t_f1), 'p_val': float(p_f1)}
    }

    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*60}")
    print(f"Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
    print(f"F1-Score: {stats['mean_f1']:.4f} ± {stats['std_f1']:.4f}")
    print(f"T-test (Accuracy): t={t_acc:.4f}, p={p_acc:.4f}")
    print(f"T-test (F1): t={t_f1:.4f}, p={p_f1:.4f}")
    print(f"{'='*60}\n")

    return results, stats

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Execute the complete Siamese Network few-shot learning pipeline:

    1. Data preparation and splitting
    2. Model training
    3. Model evaluation
    4. Visualization generation
    5. XAI explanation generation
    """
    global CLASS_NAMES

    print(f"\n{'='*70}")
    print(f"SIAMESE NETWORKS FOR FEW-SHOT LEARNING WITH XAI")
    print(f"{'='*70}\n")

    start_time = time.time()

    # Print GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.backends.cudnn.benchmark = True
    else:
        print("Running on CPU")
    print(f"{'='*70}\n")

    # ============================================
    # STEP 1: DATA SPLITTING
    # ============================================
    print("\n" + "="*50)
    print("STEP 1: DATA PREPARATION")
    print("="*50)
    df_train, df_val, df_test, CLASS_NAMES = make_stratified_splits(cfg.DATA_ROOT)

    # Visualize class distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    class_counts = df_train['class'].value_counts().sort_index()
    ax.bar(range(len(CLASS_NAMES)), class_counts.values, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels([c[:15] for c in CLASS_NAMES], rotation=45, ha='right')
    ax.set_ylabel('Number of Images')
    ax.set_title('Training Set Class Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.PLOTS_DIR, 'class_distribution.png'), dpi=150)
    plt.show()
    plt.close()

    # ============================================
    # STEP 2: MODEL TRAINING
    # ============================================
    print("\n" + "="*50)
    print("STEP 2: MODEL TRAINING (Siamese Network)")
    print("="*50)

    # Initialize model
    model = SiameseNetwork(embedding_dim=cfg.EMBEDDING_DIM).to(cfg.DEVICE)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters\n")

    # Train with OPTIMIZED function
    history, best_ckpt = train_siamese_network_optimized(
        model, df_train, df_val,
        n_epochs=cfg.EPOCHS,
        lr=cfg.LEARNING_RATE
    )

    # Plot training history
    plot_training_history(history, save_path=os.path.join(cfg.PLOTS_DIR, 'training_history.png'))

    # Save model and history
    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'siamese_final.pth'))
    with open(os.path.join(cfg.OUTPUT_DIR, 'train_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Load best checkpoint
    model.load_state_dict(torch.load(best_ckpt)['model_state'])

    # ============================================
    # STEP 3: MODEL EVALUATION
    # ============================================
    print("\n" + "="*50)
    print("STEP 3: MODEL EVALUATION")
    print("="*50)

    # Evaluate on test set
    test_metrics = evaluate_siamese_network_optimized(model, df_test)

    # Print metrics
    print_metrics(test_metrics, CLASS_NAMES)

    # Save metrics
    with open(os.path.join(cfg.OUTPUT_DIR, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)

    # Generate visualizations
    print("\nGenerating evaluation visualizations...")

    # Confusion Matrix
    plot_confusion_matrix(
        np.array(test_metrics['confusion_matrix']),
        CLASS_NAMES,
        save_path=os.path.join(cfg.PLOTS_DIR, 'confusion_matrix.png'),
        normalize=True
    )

    # Per-class metrics
    plot_per_class_metrics(
        test_metrics, CLASS_NAMES,
        save_path=os.path.join(cfg.PLOTS_DIR, 'per_class_metrics.png')
    )

    # ============================================
    # STEP 4: XAI VISUALIZATION
    # ============================================
    print("\n" + "="*50)
    print("STEP 4: XAI VISUALIZATION")
    print("="*50)

    sparsity_results = generate_xai_explanations_optimized(model, df_test, n_samples=6)

    # Save XAI results
    with open(os.path.join(cfg.OUTPUT_DIR, 'xai_results.json'), 'w') as f:
        json.dump(sparsity_results, f, indent=2)

    # ============================================
    # COMPLETION
    # ============================================
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
    # Set seed
    set_seed(cfg.RNG_SEED)

    # Create directories
    create_directories()

    # Run main pipeline
    model, test_metrics, history = main()