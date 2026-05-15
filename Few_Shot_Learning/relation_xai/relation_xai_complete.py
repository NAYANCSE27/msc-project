"""
================================================================================
RELATION NETWORKS FOR FEW-SHOT IMAGE CLASSIFICATION WITH XAI
================================================================================

A research-grade implementation of Relation Networks integrated with
Explainable AI (XAI) techniques for few-shot medical image classification.

Key Difference from Siamese Networks:
- Siamese Networks: Use fixed distance metric (e.g., Euclidean, cosine)
- Relation Networks: Learn the comparison function using a neural network
- The relation module learns to output a relation score for each (query, support) pair

Author: Research Implementation
Version: 1.0

================================================================================
MATHEMATICAL FORMULATION
================================================================================

1. EMBEDDING FUNCTION (Feature Extractor)
   f_φ: X → R^d maps images to d-dimensional embedding space

   z_i = f_φ(x_i) where x_i ∈ R^(H×W×C)

2. CONCATENATION & COMBINATION
   For comparing query x_q and support x_s:
   c = [f_φ(x_q); f_φ(x_s)]  # Concatenate embeddings
   or use element-wise product: c = f_φ(x_q) ⊙ f_φ(x_s)

3. RELATION MODULE
   g_θ: R^(2d) → R^k → R^1 maps concatenated pair to relation score

   r = g_θ([z_q; z_s])  # Learned similarity measure

4. CLASSIFICATION
   For N-way K-shot:
   - Compute relation scores between query and all support images
   - Aggregate scores for each class
   - Classify to class with highest aggregate relation score

5. LOSS FUNCTION
   Mean Squared Error (MSE) between predicted and target relation scores:
   L = (1/N) * Σ (r_pred - r_target)²

   where r_target = 1 if query and support are same class, 0 otherwise

================================================================================
EVALUATION METRICS
================================================================================

- Accuracy: (TP + TN) / (TP + TN + FP + FN)
- F1-Score (Macro): (1/C) * Σ F1_c
- ECE (Expected Calibration Error)
- Attribution Sparsity

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

# Set publication-quality plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 10

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Central configuration for the Relation Network experiment."""

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
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Model parameters
    EMBEDDING_DIM = 64     # Feature embedding dimension
    HIDDEN_DIM = 128       # Relation module hidden dimension
    IMAGE_SIZE =128

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

def make_stratified_splits(root_dir, seed=None):
    """
    Perform stratified train/val/test splitting.

    For C classes with n_c samples each:
        n_train_c = floor(n_c * 0.8)
        n_val_c = floor(n_c * 0.1)
        n_test_c = n_c - n_train_c - n_val_c
    """
    if seed is None:
        seed = cfg.RNG_SEED

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
    print(f"Images per class: {df['label'].value_counts().sort_index().tolist()}")

    x, y = df['image'], df['label']

    # First split: 80% train, 20% temp
    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=(1-cfg.TRAIN_RATIO), random_state=seed)
    train_idx, temp_idx = next(splitter1.split(x, y))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)

    # Second split: 10% val, 10% test from temp
    test_ratio_adjusted = cfg.TEST_RATIO / (cfg.VAL_RATIO + cfg.TEST_RATIO)
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio_adjusted, random_state=seed)
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

class ImagePathsDataset(Dataset):
    """Custom dataset for loading images from CSV file paths."""

    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image']).convert('RGB')
        image = transforms.ToTensor()(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, int(row['label'])

def make_transforms(img_size=cfg.IMAGE_SIZE):
    """
    Create training and evaluation transforms.

    Training augmentation includes:
    - Random horizontal flipping
    - Random rotation
    - Color jitter
    - ImageNet normalization
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, eval_transform

# =============================================================================
# RELATION NETWORK EPISODIC SAMPLER
# =============================================================================

class RelationSampler:
    """
    Sampler for creating few-shot episodes for Relation Networks.

    Each episode consists of:
    - N-way classes selected from the dataset
    - K-shot support examples per class
    - Q-query test examples per class

    For Relation Networks, we compute relation scores between each query
    and all support images to determine classification.
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

EPISODES_PER_EPOCH = 20

def episode_loader(df, transform, n_way=None, k_shot=None, q_query=None, episodes=None):
    """Helper function to create dataset and sampler."""
    if n_way is None: n_way = cfg.N_WAY
    if k_shot is None: k_shot = cfg.K_SHOT
    if q_query is None: q_query = cfg.Q_QUERY
    if episodes is None: episodes = EPISODES_PER_EPOCH

    dataset = ImagePathsDataset(df, transform=transform)
    sampler = RelationSampler(
        df['label'].to_numpy(),
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        episodes=episodes
    )
    return dataset, sampler

# =============================================================================
# RELATION NETWORK MODEL
# =============================================================================

class FeatureEncoder(nn.Module):
    """
    Convolutional encoder for extracting features from images.

    Architecture:
    Input (3, H, W)
        → ConvBlock1 (64 channels) → MaxPool (H/2)
        → ConvBlock2 (64 channels) → MaxPool (H/4)
        → ConvBlock3 (64 channels) → MaxPool (H/8)
        → ConvBlock4 (64 channels) → MaxPool (H/16)
        → AdaptiveAvgPool (4×4)
        → Output (64 channels, 4×4 spatial)

    Mathematical formulation:
    F = f_φ(x) ∈ R^(C×H'×W') where C=64
    """

    def __init__(self):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # Block 1: 3 -> 64 channels
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 64 -> 64 channels
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64 -> 64 channels
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 64 -> 64 channels
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Adaptive pooling
            nn.AdaptiveAvgPool2d((4, 4))
        )

    def forward(self, x):
        """
        Forward pass through feature encoder.

        Args:
            x: [batch_size, 3, H, W] input images

        Returns:
            features: [batch_size, 64, 4, 4] feature maps
        """
        return self.conv_blocks(x)


class RelationModule(nn.Module):
    """
    Relation module that learns to compare pairs of images.

    Takes concatenated features from query and support images and
    learns to output a relation score indicating similarity.

    Mathematical formulation:
    For query feature F_q and support feature F_s:
    1. Concatenate: C = [F_q; F_s] ∈ R^(2C×H×W)
    2. Process through CNN: R = g_θ(C)
    3. Aggregate to scalar: r = σ(MLP(R))

    Args:
        input_dim: Channels in concatenated feature (2 * feature_dim)
    """

    def __init__(self, input_dim=128):
        super().__init__()

        self.relation_network = nn.Sequential(
            # Conv layer 1
            nn.Conv2d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Conv layer 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Conv layer 3
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Flatten and MLP
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, cfg.HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(cfg.HIDDEN_DIM, 1),
            nn.Sigmoid()  # Output relation score in [0, 1]
        )

    def forward(self, x):
        """
        Forward pass through relation module.

        Args:
            x: [batch_size, 2*feature_dim, H, W] concatenated features

        Returns:
            relation_scores: [batch_size, 1] relation scores in [0, 1]
        """
        return self.model(x) if hasattr(self, 'model') else self.relation_network(x)


class RelationNetwork(nn.Module):
    """
    Relation Network for few-shot classification.

    Architecture:
    1. Feature Encoder: f_φ extracts features from images
    2. Relation Module: g_θ learns to compare query-support pairs
    3. Classification: Aggregate relation scores by class

    Forward pass:
    1. Encode query and support images: F_q = f_φ(x_q), F_s = f_φ(x_s)
    2. Concatenate features: C = [F_q; F_s]
    3. Compute relation score: r = g_θ(C)
    4. Aggregate scores by class
    5. Classify to highest aggregate score class

    Mathematical formulation:
    - Feature extraction: F = f_φ(x) ∈ R^(C×H×W)
    - Concatenation: C = [F_q; F_s] ∈ R^(2C×H×W)
    - Relation score: r = g_θ(C) ∈ [0, 1]
    - Class score: score(c) = Σ_{i: y_s^i = c} r_i
    - Prediction: y_pred = argmax_c score(c)
    """

    def __init__(self):
        super().__init__()

        # Feature encoder
        self.feature_encoder = FeatureEncoder()

        # Relation module
        self.relation_module = RelationModule(input_dim=128)

    def forward_one(self, x):
        """Extract features from a single image."""
        return self.feature_encoder(x)

    def compute_relation_scores(self, query_features, support_features):
        """
        Compute relation scores between query and support images.

        Args:
            query_features: [Q, C, H, W] query feature maps
            support_features: [S, C, H, W] support feature maps

        Returns:
            relation_scores: [Q, S] relation scores in [0, 1]
        """
        Q = query_features.size(0)
        S = support_features.size(0)

        # Reshape for batch processing
        # Expand query features to match support count
        query_expanded = query_features.unsqueeze(1).expand(Q, S, -1, -1, -1)
        support_expanded = support_features.unsqueeze(0).expand(Q, S, -1, -1, -1)

        # Concatenate along channel dimension
        # [Q, S, 2*C, H, W]
        concatenated = torch.cat([query_expanded, support_expanded], dim=2)

        # Reshape for relation module: [Q*S, 2*C, H, W]
        Q2, S2, C2, H, W = concatenated.shape
        concatenated = concatenated.reshape(Q2 * S2, C2, H, W)

        # Compute relation scores
        relation_scores = self.relation_module(concatenated)

        # Reshape back to [Q, S]
        relation_scores = relation_scores.view(Q, S)

        return relation_scores

    def forward(self, support, support_labels, query):
        """
        Forward pass for few-shot classification.

        Args:
            support: Support images [S, 3, H, W]
            support_labels: Support labels [S]
            query: Query images [Q, 3, H, W]

        Returns:
            logits: [Q, N] class scores per query
            query_features: Query features [Q, C, H, W]
            support_features: Support features [S, C, H, W]
            relation_scores: [Q, S] relation scores
        """
        # Encode all images
        support_features = self.feature_encoder(support)
        query_features = self.feature_encoder(query)

        # Compute relation scores between query and support
        relation_scores = self.compute_relation_scores(query_features, support_features)

        # Aggregate relation scores by class
        Q = query_features.size(0)
        unique_labels = torch.unique(support_labels)
        N = len(unique_labels)

        # Create label mapping
        label_map = {int(c): i for i, c in enumerate(unique_labels)}

        # Aggregate scores for each class
        logits = torch.zeros(Q, N, device=query.device)

        for idx, c in enumerate(unique_labels):
            mask = (support_labels == c).unsqueeze(0)  # [1, S]
            class_scores = (relation_scores * mask.float()).sum(dim=1)  # [Q]
            # Normalize by number of support images in class
            class_scores = class_scores / (mask.sum().float() + 1e-8)
            logits[:, idx] = class_scores

        return logits, query_features, support_features, relation_scores


class RelationLoss(nn.Module):
    """
    MSE loss for relation network training.

    For each (query, support) pair:
    - Target = 1 if same class, 0 if different class

    Mathematical formulation:
    L = (1/N) * Σ (r_pred - r_target)²

    This encourages:
    - High relation scores for same-class pairs
    - Low relation scores for different-class pairs
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, relation_scores, query_labels, support_labels):
        """
        Compute relation loss.

        Args:
            relation_scores: [Q, S] relation scores
            query_labels: [Q] query labels (mapped to 0..N-1)
            support_labels: [S] support labels (mapped to 0..N-1)

        Returns:
            loss: scalar loss value
        """
        Q, S = relation_scores.shape

        # Create target matrix [Q, S]
        # target[q, s] = 1 if query_labels[q] == support_labels[s], else 0
        query_labels_expanded = query_labels.unsqueeze(1)  # [Q, 1]
        support_labels_expanded = support_labels.unsqueeze(0)  # [1, S]
        targets = (query_labels_expanded == support_labels_expanded).float()  # [Q, S]

        # Compute MSE loss
        loss = self.mse(relation_scores, targets)

        return loss


class CrossEntropyLoss(nn.Module):
    """Standard cross-entropy loss for final classification."""

    def forward(self, logits, labels):
        return F.cross_entropy(logits, labels)

# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_ece(probs, labels, n_bins=15):
    """
    Compute Expected Calibration Error (ECE).

    ECE = Σ_{b=1}^B (|B_b|/n) * |acc(B_b) - conf(B_b)|

    Lower ECE indicates better calibration.
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

    Sparsity = (1/(H*W)) * Σ_{i,j} 1(|a_{ij}| < τ)

    where τ = 0.6 is the sparsity threshold.

    Higher sparsity indicates more focused explanations.
    """
    attr = np.abs(attributions)
    if attr.max() > 0:
        attr = attr / attr.max()

    sparsity = np.mean(attr < threshold)
    return float(sparsity)

def compute_all_metrics(y_true, y_pred, y_prob, class_names=None):
    """
    Compute comprehensive evaluation metrics.
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
# XAI METHODS FOR RELATION NETWORKS
# =============================================================================

class RelationGradCAM:
    """
    Gradient-weighted Class Activation Mapping for Relation Networks.

    For Relation Networks, we compute Grad-CAM on the query image
    with respect to its relation score to the support set.
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
        """
        self.model.eval()
        self.model.zero_grad()

        if self.support_images is None or self.support_labels is None:
            raise ValueError("GradCAM requires support_images and support_labels")

        # Prepare query image
        query_img = input_tensor.unsqueeze(0)

        # Forward pass
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


def relation_saliency_map(model, input_tensor, support_images=None, support_labels=None, target_class=None):
    """
    Compute gradient-based saliency map for Relation Network.

    The saliency map shows which pixels have the highest influence
    on the relation score computation.
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
    """Save heatmap overlay visualization."""
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

def run_relation_episode(model, optimizer, dataset, support_idx, query_idx, relation_criterion):
    """
    Run a single few-shot episode for Relation Network training.

    Training algorithm:
    1. Encode support and query images
    2. Compute relation scores between query and support
    3. Compute relation loss (MSE between scores and targets)
    4. Backpropagate and update
    """
    model.train()

    # Load data
    support_images = torch.stack([dataset[i][0] for i in support_idx]).to(cfg.DEVICE)
    support_labels = torch.tensor(
        [dataset[i][1] for i in support_idx],
        dtype=torch.long
    ).to(cfg.DEVICE)
    query_images = torch.stack([dataset[i][0] for i in query_idx]).to(cfg.DEVICE)
    query_labels = torch.tensor(
        [dataset[i][1] for i in query_idx],
        dtype=torch.long
    ).to(cfg.DEVICE)

    # Map labels to 0..N-1 for current episode
    unique = torch.unique(support_labels)
    label_map = {int(c): i for i, c in enumerate(unique)}
    support_labels_mapped = torch.tensor(
        [label_map[int(l)] for l in support_labels],
        dtype=torch.long
    ).to(cfg.DEVICE)
    query_labels_mapped = torch.tensor(
        [label_map[int(l)] for l in query_labels],
        dtype=torch.long
    ).to(cfg.DEVICE)

    # Forward pass
    logits, query_emb, support_emb, relation_scores = model(
        support_images, support_labels_mapped, query_images
    )

    # Compute relation loss
    loss = relation_criterion(relation_scores, query_labels_mapped, support_labels_mapped)

    # Also compute classification loss for better performance
    ce_loss = F.cross_entropy(logits, query_labels_mapped)
    total_loss = loss + ce_loss

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Compute metrics
    preds = torch.argmax(logits, dim=1)
    acc = (preds == query_labels_mapped).float().mean().item()
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()

    return (total_loss.item(), acc, preds.detach().cpu().numpy(),
            query_labels_mapped.detach().cpu().numpy(), probs, loss.item())


def validate_relation_episode(model, dataset, support_idx, query_idx, relation_criterion):
    """Run a single few-shot episode for validation (no gradient updates)."""
    model.eval()

    with torch.no_grad():
        support_images = torch.stack([dataset[i][0] for i in support_idx]).to(cfg.DEVICE)
        support_labels = torch.tensor(
            [dataset[i][1] for i in support_idx],
            dtype=torch.long
        ).to(cfg.DEVICE)
        query_images = torch.stack([dataset[i][0] for i in query_idx]).to(cfg.DEVICE)
        query_labels = torch.tensor(
            [dataset[i][1] for i in query_idx],
            dtype=torch.long
        ).to(cfg.DEVICE)

        # Map labels
        unique = torch.unique(support_labels)
        label_map = {int(c): i for i, c in enumerate(unique)}
        support_labels_mapped = torch.tensor(
            [label_map[int(l)] for l in support_labels],
            dtype=torch.long
        ).to(cfg.DEVICE)
        query_labels_mapped = torch.tensor(
            [label_map[int(l)] for l in query_labels],
            dtype=torch.long
        ).to(cfg.DEVICE)

        # Forward pass
        logits, query_emb, support_emb, relation_scores = model(
            support_images, support_labels_mapped, query_images
        )

        # Compute losses
        loss = relation_criterion(relation_scores, query_labels_mapped, support_labels_mapped)
        ce_loss = F.cross_entropy(logits, query_labels_mapped)
        total_loss = loss + ce_loss

        # Metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == query_labels_mapped).float().mean().item()
        probs = F.softmax(logits, dim=1).cpu().numpy()

        return (total_loss.item(), acc, preds.cpu().numpy(),
                query_labels_mapped.cpu().numpy(), probs, loss.item())


def train_relation_network(model, df_train, df_val, n_epochs=None, lr=None, verbose=True):
    """
    Complete training loop for Relation Network.

    Training algorithm:
    FOR each epoch:
        FOR each episode:
            1. Sample N classes from training set
            2. Sample K support and Q query images per class
            3. Encode images: F = f_φ(images)
            4. Compute relations: r = g_θ([F_q; F_s])
            5. Aggregate to class scores
            6. Compute loss: L = MSE(r, target) + CE(logits, labels)
            7. Update parameters: θ = θ - lr * ∇L
        END
    END
    """
    if n_epochs is None: n_epochs = cfg.EPOCHS
    if lr is None: lr = cfg.LEARNING_RATE

    # Create datasets and samplers
    train_transform, eval_transform = make_transforms()
    train_dataset, train_sampler = episode_loader(
        df_train, train_transform, episodes=EPISODES_PER_EPOCH
    )
    val_dataset, val_sampler = episode_loader(
        df_val, eval_transform, episodes=10, k_shot=cfg.K_SHOT, q_query=11
    )

    # Loss function
    relation_criterion = RelationLoss()

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )

    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_relation_loss': [],
        'val_loss': [], 'val_acc': [], 'val_relation_loss': [],
        'epoch_times': []
    }

    best_val_acc = 0.0
    best_ckpt = None

    print(f"\n{'='*60}")
    print("TRAINING RELATION NETWORK")
    print(f"{'='*60}")
    print(f"Epochs: {n_epochs}, Episodes/Epoch: {EPISODES_PER_EPOCH}")
    print(f"Learning Rate: {lr}, Weight Decay: {cfg.WEIGHT_DECAY}")
    print(f"{'='*60}\n")

    epoch_start = time.time()

    for epoch in range(1, n_epochs + 1):
        # Training episodes
        model.train()
        train_losses, train_accs, train_rel_losses = [], [], []

        for support_idx, query_idx in train_sampler:
            loss, acc, _, _, _, rel_loss = run_relation_episode(
                model, optimizer, train_dataset, support_idx, query_idx, relation_criterion
            )
            train_losses.append(loss)
            train_accs.append(acc)
            train_rel_losses.append(rel_loss)

        scheduler.step()

        # Validation episodes
        model.eval()
        val_losses, val_accs, val_rel_losses = [], [], []

        with torch.no_grad():
            for support_idx, query_idx in val_sampler:
                loss, acc, _, _, _, rel_loss = validate_relation_episode(
                    model, val_dataset, support_idx, query_idx, relation_criterion
                )
                val_losses.append(loss)
                val_accs.append(acc)
                val_rel_losses.append(rel_loss)

        # Compute epoch statistics
        train_loss = float(np.mean(train_losses))
        train_acc = float(np.mean(train_accs))
        train_rel_loss = float(np.mean(train_rel_losses))
        val_loss = float(np.mean(val_losses))
        val_acc = float(np.mean(val_accs))
        val_rel_loss = float(np.mean(val_rel_losses))
        epoch_time = time.time() - epoch_start

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_relation_loss'].append(train_rel_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_relation_loss'].append(val_rel_loss)
        history['epoch_times'].append(epoch_time)

        if verbose and epoch % 2 == 0:
            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.3f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.3f} | "
                f"Time: {epoch_time:.1f}s"
            )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt = os.path.join(cfg.CKPT_DIR, 'best_relation.pth')
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

def evaluate_relation_network(model, df_test, episodes=20):
    """
    Evaluate Relation Network on test set with episodic sampling.
    """
    _, eval_transform = make_transforms()
    test_dataset, test_sampler = episode_loader(
        df_test, eval_transform,
        episodes=episodes,
        k_shot=cfg.K_SHOT,
        q_query=11
    )

    relation_criterion = RelationLoss()

    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    all_loss = []
    all_acc = []

    with torch.no_grad():
        for support_idx, query_idx in test_sampler:
            # Load data
            support_images = torch.stack([test_dataset[i][0] for i in support_idx]).to(cfg.DEVICE)
            support_labels = torch.tensor(
                [test_dataset[i][1] for i in support_idx],
                dtype=torch.long
            ).to(cfg.DEVICE)
            query_images = torch.stack([test_dataset[i][0] for i in query_idx]).to(cfg.DEVICE)
            query_labels = torch.tensor(
                [test_dataset[i][1] for i in query_idx],
                dtype=torch.long
            ).to(cfg.DEVICE)

            # Map labels
            unique = torch.unique(support_labels)
            label_map = {int(c): i for i, c in enumerate(unique)}
            support_labels_mapped = torch.tensor(
                [label_map[int(l)] for l in support_labels],
                dtype=torch.long
            ).to(cfg.DEVICE)
            query_labels_mapped = torch.tensor(
                [label_map[int(l)] for l in query_labels],
                dtype=torch.long
            ).to(cfg.DEVICE)

            # Forward pass
            logits, query_emb, support_emb, relation_scores = model(
                support_images, support_labels_mapped, query_images
            )

            # Compute loss and predictions
            loss = relation_criterion(relation_scores, query_labels_mapped, support_labels_mapped)
            ce_loss = F.cross_entropy(logits, query_labels_mapped)
            total_loss = loss + ce_loss

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            probs = F.softmax(logits, dim=1).cpu().numpy()

            # Accumulate results
            all_loss.append(total_loss.item())
            all_acc.append((preds == query_labels_mapped.cpu().numpy()).mean())

            y_true.extend(query_labels_mapped.cpu().numpy().tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())

    # Compute metrics
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

    # Total loss curves
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss (Total)')
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

    # Relation loss curves
    axes[1, 0].plot(epochs, history['train_relation_loss'], 'b-', label='Train Rel Loss', linewidth=2)
    axes[1, 0].plot(epochs, history['val_relation_loss'], 'r-', label='Val Rel Loss', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Relation Loss (MSE)')
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
    else:
        cm_normalized = cm
        fmt = 'd'

    sns.heatmap(cm_normalized, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': 'Proportion'})

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

def generate_xai_explanations(model, df_test, n_samples=5, save_dir=None):
    """
    Generate XAI explanations for test samples using Relation Network.
    """
    if save_dir is None: save_dir = cfg.XAI_DIR

    _, eval_transform = make_transforms()
    ds = ImagePathsDataset(df_test, transform=eval_transform)
    sampler = RelationSampler(
        df_test['label'].to_numpy(),
        n_way=cfg.N_WAY, k_shot=cfg.K_SHOT, q_query=11, episodes=1
    )

    support_idx, query_idx = next(iter(sampler))

    # Load support set
    support_images = torch.stack([ds[i][0] for i in support_idx]).to(cfg.DEVICE)
    support_labels = torch.tensor(
        [ds[i][1] for i in support_idx],
        dtype=torch.long
    ).to(cfg.DEVICE)

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
    print("XAI VISUALIZATION GENERATION (Relation Network)")
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

        # Target layer for Grad-CAM
        target_layer = model.feature_encoder.conv_blocks[4]

        # Generate Grad-CAM
        gradcam = RelationGradCAM(
            model,
            target_layer=target_layer,
            support_images=support_images,
            support_labels=support_labels_mapped
        )
        cam_mask = gradcam.generate(img, target_class=pred)
        gradcam.close()

        # Generate saliency map
        sal_map = relation_saliency_map(
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
            os.path.join(save_dir, f'relation_gradcam_sample{idx}_true{true_label}_pred{pred_original}.png'),
            title=f'Relation Grad-CAM: True={true_label}, Pred={pred_original}'
        )

        save_heatmap(
            img.cpu(), sal_map,
            os.path.join(save_dir, f'relation_saliency_sample{idx}_true{true_label}_pred{pred_original}.png'),
            title=f'Relation Saliency: True={true_label}, Pred={pred_original}'
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
    """
    if epochs_per_run is None:
        epochs_per_run = max(5, cfg.EPOCHS // 3)

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
        model = RelationNetwork().to(cfg.DEVICE)

        # Train
        history, ckpt = train_relation_network(
            model, df_train, df_val,
            n_epochs=epochs_per_run,
            lr=cfg.LEARNING_RATE,
            verbose=False
        )

        # Load best checkpoint
        model.load_state_dict(torch.load(ckpt)['model_state'])

        # Evaluate
        metrics = evaluate_relation_network(model, df_test)
        metrics['run'] = run
        metrics['final_train_acc'] = history['train_acc'][-1]
        metrics['final_val_acc'] = history['val_acc'][-1]

        results.append(metrics)

        print(f"  Run {run} Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Run {run} Test F1-Macro: {metrics['f1_macro']:.4f}")

    # Statistical analysis
    accuracies = [r['accuracy'] for r in results]
    f1_scores = [r['f1_macro'] for r in results]

    # T-test
    t_acc, p_acc = ttest_ind([accuracies[0]], accuracies[1:] if len(accuracies) > 1 else accuracies)
    t_f1, p_f1 = ttest_ind([f1_scores[0]], f1_scores[1:] if len(f1_scores) > 1 else f1_scores)

    stats = {
        'n_runs': n_runs,
        'accuracies': accuracies,
        'f1_scores': f1_scores,
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
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
    Execute the complete Relation Network few-shot learning pipeline:

    1. Data preparation and splitting
    2. Model training
    3. Model evaluation
    4. Visualization generation
    5. XAI explanation generation
    """
    global CLASS_NAMES

    print(f"\n{'='*70}")
    print(f"RELATION NETWORKS FOR FEW-SHOT LEARNING WITH XAI")
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

    # STEP 1: DATA SPLITTING
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

    # STEP 2: MODEL TRAINING
    print("\n" + "="*50)
    print("STEP 2: MODEL TRAINING (Relation Network)")
    print("="*50)

    # Initialize model
    model = RelationNetwork().to(cfg.DEVICE)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters\n")

    # Train
    history, best_ckpt = train_relation_network(
        model, df_train, df_val,
        n_epochs=cfg.EPOCHS,
        lr=cfg.LEARNING_RATE
    )

    # Plot training history
    plot_training_history(history, save_path=os.path.join(cfg.PLOTS_DIR, 'training_history.png'))

    # Save model and history
    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'relation_final.pth'))
    with open(os.path.join(cfg.OUTPUT_DIR, 'train_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Load best checkpoint
    model.load_state_dict(torch.load(best_ckpt)['model_state'])

    # STEP 3: MODEL EVALUATION
    print("\n" + "="*50)
    print("STEP 3: MODEL EVALUATION")
    print("="*50)

    # Evaluate on test set
    test_metrics = evaluate_relation_network(model, df_test)

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

    # STEP 4: XAI VISUALIZATION
    print("\n" + "="*50)
    print("STEP 4: XAI VISUALIZATION")
    print("="*50)

    sparsity_results = generate_xai_explanations(model, df_test, n_samples=6)

    # Save XAI results
    with open(os.path.join(cfg.OUTPUT_DIR, 'xai_results.json'), 'w') as f:
        json.dump(sparsity_results, f, indent=2)

    # COMPLETION
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