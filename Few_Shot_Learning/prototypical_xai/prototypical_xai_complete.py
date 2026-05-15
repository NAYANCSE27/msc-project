"""
================================================================================
PROTOTYPICAL NETWORKS FOR FEW-SHOT IMAGE CLASSIFICATION WITH XAI
================================================================================

A research-grade implementation of Prototypical Networks integrated with
Explainable AI (XAI) techniques for few-shot medical image classification.

Author: Research Implementation
Version: 1.0

================================================================================
MATHEMATICAL FORMULATION
================================================================================

1. EMBEDDING FUNCTION
   f_φ: X → R^d maps images to d-dimensional embedding space

   z_i = f_φ(x_i) where x_i ∈ R^(H×W×C)

2. PROTOTYPE COMPUTATION
   For each class c, compute prototype as mean of support embeddings:

   p_c = (1/|S_c|) * Σ_{x_i ∈ S_c} f_φ(x_i)

   where S_c is the support set for class c

3. DISTANCE METRIC (Squared Euclidean)
   d(z, p_c) = ||z - p_c||² = Σ_{j=1}^d (z_j - p_{c,j})²

4. CLASSIFICATION LOGITS
   ℓ_c = -d(f_φ(x), p_c)

   P(y=c|x) = exp(ℓ_c) / Σ_{c'} exp(ℓ_{c'})

5. LOSS FUNCTION (Negative Log-Likelihood)
   L = -(1/|Q|) * Σ_{(x,y)∈Q} log P(y|x)

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
from torch.utils.data import Dataset, DataLoader
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
    """Central configuration for the experiment."""

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
    EPISODES_PER_EPOCH = 20
    VAL_EPISODES = 10
    TEST_EPISODES = 20
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4

    # Model parameters
    EMBEDDING_DIM = 128
    IMAGE_SIZE = 128

cfg = Config()

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    print(f"Images per class: {df['label'].value_counts().sort_index().tolist()}")

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
    - Random horizontal flipping for horizontal invariance
    - Random rotation for rotation robustness
    - Color jitter for lighting invariance
    - ImageNet normalization for transfer learning compatibility
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
# EPISODIC SAMPLING
# =============================================================================

class EpisodicSampler:
    """
    Sampler for creating few-shot episodes.

    Each episode consists of:
    - N-way classes selected from the dataset
    - K-shot support examples per class
    - Q-query test examples per class

    Episode sampling algorithm:
    1. Randomly select N classes without replacement
    2. For each class, sample K+Q examples without replacement
    3. Split into support (K) and query (Q) sets

    Mathematical formulation:
    For episode e:
        C_e = {c_1, c_2, ..., c_N} sampled without replacement
        S_c = {x_1, ..., x_K} support set for class c
        Q_c = {x_{K+1}, ..., x_{K+Q}} query set for class c
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

def exemplar_loader(df, transform, n_way=None, k_shot=None, q_query=None, episodes=None):
    """Helper function to create dataset and sampler."""
    if n_way is None: n_way = cfg.N_WAY
    if k_shot is None: k_shot = cfg.K_SHOT
    if q_query is None: q_query = cfg.Q_QUERY
    if episodes is None: episodes = cfg.EPISODES_PER_EPOCH

    dataset = ImagePathsDataset(df, transform=transform)
    sampler = EpisodicSampler(
        df['label'].to_numpy(),
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        episodes=episodes
    )
    return dataset, sampler

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class ConvEncoder(nn.Module):
    """
    Convolutional encoder for embedding images into vector space.

    Architecture:
    Input (3, H, W)
        → ConvBlock1 (32 channels) → MaxPool (H/2)
        → ConvBlock2 (64 channels) → MaxPool (H/4)
        → ConvBlock3 (128 channels) → MaxPool (H/8)
        → ConvBlock4 (128 channels) → MaxPool (H/16)
        → AdaptiveAvgPool (2×2)
        → FC (256 → 128)
        → Output (128-dim embedding)

    Mathematical formulation:
    z = f_φ(x) = W_2 * ReLU(W_1 * h + b_1) + b_2
    where h is the pooled conv features
    """

    def __init__(self, out_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Adaptive pooling to fixed size
            nn.AdaptiveAvgPool2d((2, 2))
        )

        # Projection head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

class ProtoNet(nn.Module):
    """
    Prototypical Network for few-shot classification.

    Forward pass:
    1. z_support = f_phi(support) - embed support images
    2. z_query = f_phi(query) - embed query images
    3. p_c = mean(z_support for class c) - compute prototypes
    4. d(x, p_c) = ||z_query - p_c||² - compute distances
    5. logits = -d(x, p_c) - convert to logits

    Mathematical formulation:
    - Prototype: p_c = (1/|S_c|) * Σ_{x∈S_c} f_φ(x)
    - Distance: d(z, p_c) = ||z - p_c||²
    - Logits: ℓ_c = -d(f_φ(x), p_c)
    - Probability: P(y=c|x) = softmax(ℓ)_c
    """

    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, support, support_labels, query):
        """
        Forward pass through prototypical network.

        Args:
            support: Support images [N*K, C, H, W]
            support_labels: Support labels [N*K]
            query: Query images [N*Q, C, H, W]

        Returns:
            logits: Classification logits [N*Q, N]
            prototypes: Class prototypes [N, D]
            z_query: Query embeddings [N*Q, D]
        """
        # Encode support and query images
        z_support = self.embedding_net(support)
        z_query = self.embedding_net(query)

        # Compute prototypes per class
        unique_labels = torch.unique(support_labels)
        prototypes = []
        for c in unique_labels:
            class_embeddings = z_support[support_labels == c]
            prototypes.append(class_embeddings.mean(dim=0))
        prototypes = torch.stack(prototypes)

        # Compute squared Euclidean distances to prototypes
        dists = euclidean_dist(z_query, prototypes)

        # Convert distances to logits
        logits = -dists

        return logits, prototypes, z_query

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

def proto_loss(logits, query_labels):
    """
    Compute prototypical loss (negative log-likelihood).

    Mathematical formula:
    L = -log P(y=c|x) = -log [exp(-d(x,p_c)) / Σ_c' exp(-d(x,p_c'))]

    This is equivalent to cross-entropy loss on the distance-based logits.

    Args:
        logits: [N*Q, N] classification logits
        query_labels: [N*Q] ground truth labels (0 to N-1)

    Returns:
        loss: Scalar loss value
    """
    return F.cross_entropy(logits, query_labels)

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
# XAI METHODS
# =============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for Prototypical Networks.

    Grad-CAM generates visual explanations by using the gradients
    that flow into the target convolutional layer to produce
    a saliency map highlighting important regions.

    Mathematical formulation:
    L_Grad-CAM^(c) = ReLU(Σ_k α_k^c * A^k)

    where:
    - A^k is the activation map of the k-th convolutional feature map
    - α_k^c = (1/Z) * Σ_i Σ_j (∂y^c / ∂A_{ij}^k) is the gradient-weighted weight
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

        # Forward pass
        output, _, _ = self.model(self.support_images, self.support_labels, query_img)

        # Get target class
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        if target_class >= output.shape[1]:
            target_class = torch.argmax(output, dim=1).item()

        # Backward pass
        score = output[0, target_class]
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

def saliency_map(model, input_tensor, support_images=None, support_labels=None, target_class=None):
    """
    Compute gradient-based saliency map.

    The saliency map shows which pixels have the highest influence
    on the prediction for the target class.

    Mathematical formulation:
    S_{ij} = |∂y^c / ∂x_{ij}|

    where y^c is the score for class c and x_{ij} is the pixel at position (i,j).

    Args:
        model: ProtoNet model
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
    logits, _, _ = model(support_images, support_labels, input_tensor)

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

def run_episode(model, optimizer, dataset, support_idx, query_idx):
    """Run a single few-shot episode for training."""
    model.train()

    # Load data
    support_images = torch.stack([dataset[i][0] for i in support_idx]).to(DEVICE)
    support_labels = torch.tensor(
        [dataset[i][1] for i in support_idx],
        dtype=torch.long
    ).to(DEVICE)
    query_images = torch.stack([dataset[i][0] for i in query_idx]).to(DEVICE)
    query_labels = torch.tensor(
        [dataset[i][1] for i in query_idx],
        dtype=torch.long
    ).to(DEVICE)

    # Map labels to 0..N-1 for current episode
    unique = torch.unique(support_labels)
    label_map = {int(c): i for i, c in enumerate(unique)}
    support_labels_mapped = torch.tensor(
        [label_map[int(l)] for l in support_labels],
        dtype=torch.long
    ).to(DEVICE)
    query_labels_mapped = torch.tensor(
        [label_map[int(l)] for l in query_labels],
        dtype=torch.long
    ).to(DEVICE)

    # Forward pass
    logits, prototypes, z_query = model(support_images, support_labels_mapped, query_images)

    # Compute loss
    loss = proto_loss(logits, query_labels_mapped)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute metrics
    preds = torch.argmax(logits, dim=1)
    acc = (preds == query_labels_mapped).float().mean().item()
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()

    return (loss.item(), acc, preds.detach().cpu().numpy(),
            query_labels_mapped.detach().cpu().numpy(), probs)

def validate_episode(model, dataset, support_idx, query_idx):
    """Run a single few-shot episode for validation (no gradient updates)."""
    model.eval()

    with torch.no_grad():
        # Load data
        support_images = torch.stack([dataset[i][0] for i in support_idx]).to(DEVICE)
        support_labels = torch.tensor(
            [dataset[i][1] for i in support_idx],
            dtype=torch.long
        ).to(DEVICE)
        query_images = torch.stack([dataset[i][0] for i in query_idx]).to(DEVICE)
        query_labels = torch.tensor(
            [dataset[i][1] for i in query_idx],
            dtype=torch.long
        ).to(DEVICE)

        # Map labels
        unique = torch.unique(support_labels)
        label_map = {int(c): i for i, c in enumerate(unique)}
        support_labels_mapped = torch.tensor(
            [label_map[int(l)] for l in support_labels],
            dtype=torch.long
        ).to(DEVICE)
        query_labels_mapped = torch.tensor(
            [label_map[int(l)] for l in query_labels],
            dtype=torch.long
        ).to(DEVICE)

        # Forward pass
        logits, prototypes, z_query = model(support_images, support_labels_mapped, query_images)

        # Compute loss
        loss = proto_loss(logits, query_labels_mapped)

        # Metrics
        preds = torch.argmax(logits, dim=1)
        acc = (preds == query_labels_mapped).float().mean().item()
        probs = F.softmax(logits, dim=1).cpu().numpy()

        return (loss.item(), acc, preds.cpu().numpy(),
                query_labels_mapped.cpu().numpy(), probs)

def train_protonet(model, df_train, df_val, n_epochs=None, lr=None, verbose=True):
    """
    Complete training loop for Prototypical Network.

    Training algorithm:
    FOR each epoch:
        FOR each episode:
            1. Sample N classes from training set
            2. Sample K support and Q query images per class
            3. Compute embeddings: z_support, z_query = f_phi(images)
            4. Compute prototypes: p_c = mean(z_support for class c)
            5. Compute distances: d_i = ||z_query - p_c||²
            6. Compute logits: l_c = -d_i
            7. Compute loss: L = CrossEntropy(l, y_query)
            8. Update parameters: θ = θ - lr * ∇L
        END
    END

    Args:
        model: ProtoNet model
        df_train: training DataFrame
        df_val: validation DataFrame
        n_epochs: number of training epochs
        lr: learning rate

    Returns:
        history: training history dict
        best_ckpt: path to best model checkpoint
    """
    if n_epochs is None: n_epochs = cfg.NUM_EPOCHS
    if lr is None: lr = cfg.LEARNING_RATE

    # Create datasets and samplers
    train_transform, eval_transform = make_transforms()
    train_dataset, train_sampler = exemplar_loader(
        df_train, train_transform, episodes=cfg.EPISODES_PER_EPOCH
    )
    val_dataset, val_sampler = exemplar_loader(
        df_val, eval_transform, episodes=cfg.VAL_EPISODES,
        k_shot=cfg.K_SHOT, q_query=11
    )

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
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'epoch_times': []
    }

    best_val_acc = 0.0
    best_ckpt = None

    print(f"\n{'='*60}")
    print("TRAINING PROTOTYPICAL NETWORK")
    print(f"{'='*60}")
    print(f"Epochs: {n_epochs}, Episodes/Epoch: {cfg.EPISODES_PER_EPOCH}")
    print(f"Learning Rate: {lr}, Weight Decay: {cfg.WEIGHT_DECAY}")
    print(f"{'='*60}\n")

    epoch_start = time.time()

    for epoch in range(1, n_epochs + 1):
        # Training episodes
        model.train()
        train_losses = []
        train_accs = []

        for support_idx, query_idx in train_sampler:
            loss, acc, _, _, _ = run_episode(
                model, optimizer, train_dataset, support_idx, query_idx
            )
            train_losses.append(loss)
            train_accs.append(acc)

        scheduler.step()

        # Validation episodes
        model.eval()
        val_losses = []
        val_accs = []

        with torch.no_grad():
            for support_idx, query_idx in val_sampler:
                loss, acc, _, _, _ = validate_episode(
                    model, val_dataset, support_idx, query_idx
                )
                val_losses.append(loss)
                val_accs.append(acc)

        # Compute epoch statistics
        train_loss = float(np.mean(train_losses))
        train_acc = float(np.mean(train_accs))
        val_loss = float(np.mean(val_losses))
        val_acc = float(np.mean(val_accs))
        epoch_time = time.time() - epoch_start

        # Record history
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

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt = os.path.join(cfg.CKPT_DIR, 'best_protonet.pth')
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

def evaluate_protonet(model, df_test, episodes=None):
    """
    Evaluate model on test set with episodic sampling.

    Args:
        model: trained ProtoNet
        df_test: test DataFrame
        episodes: number of test episodes

    Returns:
        metrics: comprehensive evaluation metrics
    """
    if episodes is None: episodes = cfg.TEST_EPISODES

    _, eval_transform = make_transforms()
    test_dataset, test_sampler = exemplar_loader(
        df_test, eval_transform,
        episodes=episodes,
        k_shot=cfg.K_SHOT,
        q_query=11
    )

    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    all_loss = []
    all_acc = []

    with torch.no_grad():
        for support_idx, query_idx in test_sampler:
            # Load data
            support_images = torch.stack([test_dataset[i][0] for i in support_idx]).to(DEVICE)
            support_labels = torch.tensor(
                [test_dataset[i][1] for i in support_idx],
                dtype=torch.long
            ).to(DEVICE)
            query_images = torch.stack([test_dataset[i][0] for i in query_idx]).to(DEVICE)
            query_labels = torch.tensor(
                [test_dataset[i][1] for i in query_idx],
                dtype=torch.long
            ).to(DEVICE)

            # Map labels
            unique = torch.unique(support_labels)
            label_map = {int(c): i for i, c in enumerate(unique)}
            support_labels_mapped = torch.tensor(
                [label_map[int(l)] for l in support_labels],
                dtype=torch.long
            ).to(DEVICE)
            query_labels_mapped = torch.tensor(
                [label_map[int(l)] for l in query_labels],
                dtype=torch.long
            ).to(DEVICE)

            # Forward pass
            logits, _, _ = model(support_images, support_labels_mapped, query_images)

            # Compute loss and predictions
            loss = proto_loss(logits, query_labels_mapped)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            probs = F.softmax(logits, dim=1).cpu().numpy()

            # Accumulate results
            all_loss.append(loss.item())
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
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
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

def generate_xai_explanations(model, df_test, n_samples=5, save_dir=None):
    """
    Generate XAI explanations for test samples.

    For each sample:
    1. Get support set from episode
    2. Compute Grad-CAM visualization
    3. Compute saliency map
    4. Save visualizations
    """
    if save_dir is None: save_dir = cfg.XAI_DIR

    _, eval_transform = make_transforms()
    ds = ImagePathsDataset(df_test, transform=eval_transform)
    sampler = EpisodicSampler(
        df_test['label'].to_numpy(),
        n_way=cfg.N_WAY, k_shot=cfg.K_SHOT, q_query=11, episodes=1
    )

    support_idx, query_idx = next(iter(sampler))

    # Load support set
    support_images = torch.stack([ds[i][0] for i in support_idx]).to(DEVICE)
    support_labels = torch.tensor(
        [ds[i][1] for i in support_idx],
        dtype=torch.long
    ).to(DEVICE)

    # Map labels
    unique = torch.unique(support_labels)
    label_map = {int(c): i for i, c in enumerate(unique)}
    support_labels_mapped = torch.tensor(
        [label_map[int(l)] for l in support_labels],
        dtype=torch.long
    ).to(DEVICE)

    # Get query images and labels
    query_images = torch.stack([ds[i][0] for i in query_idx]).to(DEVICE)
    query_labels = torch.tensor([ds[i][1] for i in query_idx], dtype=torch.long).to(DEVICE)

    print(f"\n{'='*60}")
    print("XAI VISUALIZATION GENERATION")
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
        logits, _, _ = model(support_images, support_labels_mapped, img.unsqueeze(0))
        pred = torch.argmax(logits, dim=1).item()
        confidence = F.softmax(logits, dim=1)[0, pred].item()

        # Map prediction back to original label
        pred_original = int(list(label_map.keys())[list(label_map.values()).index(pred)])

        # Target layer for Grad-CAM
        target_layer = model.embedding_net.encoder[4]

        # Generate Grad-CAM
        gradcam = GradCAM(
            model,
            target_layer=target_layer,
            support_images=support_images,
            support_labels=support_labels_mapped
        )
        cam_mask = gradcam.generate(img, target_class=pred)
        gradcam.close()

        # Generate saliency map
        sal_map = saliency_map(
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
            os.path.join(save_dir, f'gradcam_sample{idx}_true{true_label}_pred{pred_original}.png'),
            title=f'Grad-CAM: True={true_label}, Pred={pred_original}, Conf={confidence:.2f}'
        )

        save_heatmap(
            img.cpu(), sal_map,
            os.path.join(save_dir, f'saliency_sample{idx}_true{true_label}_pred{pred_original}.png'),
            title=f'Saliency: True={true_label}, Pred={pred_original}, Conf={confidence:.2f}'
        )

        print(f"Sample {idx}: True={true_label}, Pred={pred_original}, "
              f"Conf={confidence:.3f}, CAM sparsity={sparsity_cam:.3f}, "
              f"Saliency sparsity={sparsity_sal:.3f}")

    return sparsity_results

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Execute the complete few-shot learning pipeline:

    1. Data preparation and splitting
    2. Model training
    3. Model evaluation
    4. Visualization generation
    5. XAI explanation generation
    """
    global CLASS_NAMES

    print(f"\n{'='*70}")
    print(f"PROTOTYPICAL NETWORKS FOR FEW-SHOT LEARNING WITH XAI")
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
    print("STEP 2: MODEL TRAINING")
    print("="*50)

    # Initialize model
    model = ProtoNet(ConvEncoder(out_dim=cfg.EMBEDDING_DIM)).to(DEVICE)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters\n")

    # Train
    history, best_ckpt = train_protonet(model, df_train, df_val)

    # Plot training history
    plot_training_history(history, save_path=os.path.join(cfg.PLOTS_DIR, 'training_history.png'))

    # Save model and history
    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'protonet_final.pth'))
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
    test_metrics = evaluate_protonet(model, df_test)

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

    sparsity_results = generate_xai_explanations(model, df_test, n_samples=6)

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