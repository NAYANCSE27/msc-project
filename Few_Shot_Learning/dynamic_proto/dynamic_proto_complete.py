"""
Dynamic Prototype FSL: Few-Shot Learning with Dynamic Prototype Refinement
=========================================================================

This module implements Dynamic Prototype Methods for Few-Shot Learning
integrated with Explainable AI (XAI) techniques.

Key Innovation: Unlike static prototypical networks, Dynamic Prototype Methods
iteratively refine class prototypes by considering query-to-prototype relationships,
enabling transductive inference and adaptive prototype computation.

Author: NAYANCSE27
Research Level Implementation
"""

import os
import json
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from scipy import stats
from scipy.stats import ttest_ind, wilcoxon
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, balanced_accuracy_score
)
from sklearn.calibration import calibration_curve

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

from captum.attr import Saliency, IntegratedGradients

CONFIG = {
    'seed': 42,
    'data_root': './data',
    'output_dir': './dynamic_proto_output',
    'num_classes': 8,
    'images_per_class': 160,
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'n_way': 5,
    'k_shot': 5,
    'n_query': 15,
    'episodes': 1500,
    'hidden_dim': 256,
    'proto_dim': 128,
    'refinement_steps': 3,
    'dropout': 0.3,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'batch_size': 16,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'patience': 25,
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CONFIG['seed'])

# =============================================================================
# MATHEMATICAL FORMULATION
# =============================================================================
"""
DYNAMIC PROTOTYPE FSL - MATHEMATICAL FRAMEWORK
==============================================

1. FEATURE ENCODING
   Given an image x_i, we extract features using a CNN encoder f_θ:

   h_i = f_θ(x_i) ∈ ℝ^d

   The encoder is trained to produce discriminative embeddings.

2. INITIAL PROTOTYPE COMPUTATION
   Standard mean-based prototype computation from support set:

   c_k^(0) = (1/|S_k|) Σ_{x_i∈S_k} h_i

   where S_k is the support set for class k.

3. DYNAMIC PROTOTYPE REFINEMENT (Key Innovation)
   Unlike static prototypes, we iteratively refine prototypes by:

   a) Computing query-to-prototype similarity:
      α_{qk} = softmax_k(s^T(q) · c_k^(t-1))

   b) Computing query-to-sample attention within each class:
      β_{ik} = softmax_i(exp(-d(h_q, h_i)))  for x_i ∈ S_k

   c) Refining prototypes with query information:
      c_k^(t) = γ_k · c_k^(t-1) + (1 - γ_k) · Σ_i β_{ik} · h_i

   where t = 1, 2, ..., T (refinement steps)
   and γ_k is a learnable decay parameter per class.

4. ALTERNATIVE: TRANSDUCTIVE PROTOTYPE UPDATE
   Using both support and query statistics:

   c_k^(t) = (1/|S_k|+|Q_k|) × [Σ_{x_i∈S_k} h_i + Σ_{x_q∈Q_k} w_q·h_q]

   where weights w_q are computed based on initial prototype similarity.

5. QUERY CLASSIFICATION
   After T refinement steps:

   P(y_q = k | x_q) = softmax_k(-d(h_q, c_k^(T)))

   Classification logit:
   s_k = -||h_q - c_k^(T)||²

6. LOSS FUNCTION
   Episode loss (negative log-probability):

   L_episode = -Σ_{q∈Q} log P(y_q | x_q, S)

   = -Σ_{q∈Q} log softmax_k(-d(h_q, c_k^(T)))[y_q]

7. ATTENTION-BASED PROTOTYPE AGGREGATION
   Instead of simple mean, use attention-weighted aggregation:

   c_k = Σ_i a_i · h_i / Σ_i a_i

   where a_i = attention(h_q, h_i) = v^T · tanh(W·[h_q; h_i])

8. EVALUATION METRICS

   Accuracy: Acc = (TP + TN) / (TP + TN + FP + FN)

   F1-Score: F1 = 2 × (Precision × Recall) / (Precision + Recall)

   Expected Calibration Error:
   ECE = Σ_b |B_b|/n × |acc(B_b) - conf(B_b)|

   Attribution Sparsity:
   Sparsity = 1 - (non_zero_attributions / total_attributions)
"""

# =============================================================================
# DATASET IMPLEMENTATION
# =============================================================================

class FSLDataset(Dataset):
    """Few-Shot Learning Dataset with stratified sampling."""

    def __init__(self, root_dir, split='train', transform=None, config=CONFIG):
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.transform = transform
        self.config = config

        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}

        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
            for img_path in class_dir.glob('*.png'):
                self.samples.append((str(img_path), self.class_to_idx[class_name]))
            for img_path in class_dir.glob('*.jpeg'):
                self.samples.append((str(img_path), self.class_to_idx[class_name]))

        self.class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(self.samples):
            self.class_indices[label].append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = torchvision.io.read_image(img_path)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        image = image.float() / 255.0

        if image.shape[1] != 84 or image.shape[2] != 84:
            image = F.interpolate(image.unsqueeze(0), size=(84, 84),
                                  mode='bilinear', align_corners=False).squeeze(0)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_samples(self, class_idx, num_samples=None):
        indices = self.class_indices[class_idx]
        if num_samples is not None:
            indices = random.sample(indices, min(num_samples, len(indices)))
        return [self.samples[i] for i in indices]


def create_stratified_split(root_dir, output_dir, config=CONFIG):
    """Create stratified train/val/test split."""
    import shutil

    output_dir = Path(output_dir)
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in Path(root_dir).iterdir() if d.is_dir()])

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = (list(class_dir.glob('*.jpg')) +
                  list(class_dir.glob('*.png')) +
                  list(class_dir.glob('*.jpeg')))
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * config['train_ratio'])
        n_val = int(n_total * config['val_ratio'])

        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        for split_name, split_images in splits.items():
            split_dir = output_dir / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for img_path in split_images:
                dest_path = split_dir / img_path.name
                if not dest_path.exists():
                    shutil.copy(img_path, dest_path)

    print(f"Dataset split created in {output_dir}")
    return output_dir


# =============================================================================
# MODEL ARCHITECTURE: DYNAMIC PROTOTYPE FSL
# =============================================================================

class ConvEncoder(nn.Module):
    """
    CNN Feature Encoder for Dynamic Prototype FSL.

    Architecture: 4 convolutional blocks with progressive channel increase.
    Each block: Conv → BatchNorm → ReLU → MaxPool
    Output: 256-dimensional embedding
    """

    def __init__(self, in_channels=3, hidden_dim=256):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            self._make_conv_block(in_channels, 64, kernel_size=3, padding=1),
            self._make_conv_block(64, 64, kernel_size=3, padding=1),
            self._make_conv_block(64, 128, kernel_size=3, padding=1),
            self._make_conv_block(128, 128, kernel_size=3, padding=1),
            self._make_conv_block(128, hidden_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.out_features = hidden_dim * 4 * 4

    def _make_conv_block(self, in_ch, out_ch, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv_blocks(x).view(x.size(0), -1)


class PrototypeAttention(nn.Module):
    """
    Attention mechanism for dynamic prototype computation.

    Computes attention weights between query and support samples
    to enable adaptive prototype aggregation.

    Attention computation:
    a_ij = v^T · tanh(W · [h_i; h_j])

    where [h_i; h_j] is the concatenation of two feature vectors.
    """

    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()

        self.W = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, support_embeddings, query_embeddings):
        """
        Args:
            support_embeddings: [n_support, feature_dim]
            query_embeddings: [n_query, feature_dim]

        Returns:
            attention_weights: [n_query, n_support] attention matrix
        """
        n_support = support_embeddings.size(0)
        n_query = query_embeddings.size(0)

        support_expanded = support_embeddings.unsqueeze(0).expand(n_query, -1, -1)
        query_expanded = query_embeddings.unsqueeze(1).expand(-1, n_support, -1)

        combined = torch.cat([support_expanded, query_expanded], dim=-1)

        attention_scores = self.W(combined).squeeze(-1)

        attention_weights = F.softmax(attention_scores, dim=-1)

        return attention_weights


class DynamicPrototypeRefiner(nn.Module):
    """
    Dynamic Prototype Refinement Module.

    Key Innovation: Iteratively refines prototypes by considering query information.
    This enables the model to adapt prototypes to the current episode's distribution.

    The refinement process:
    1. Start with initial prototype (mean of support embeddings)
    2. For each refinement step:
       - Compute attention weights between queries and supports
       - Aggregate query-aware information into prototypes
       - Update prototypes with learnable decay
    """

    def __init__(self, feature_dim, proto_dim=128, num_refinement_steps=3, dropout=0.3):
        super().__init__()

        self.feature_dim = feature_dim
        self.proto_dim = proto_dim
        self.num_steps = num_refinement_steps

        self.proto_projection = nn.Linear(feature_dim, proto_dim)

        self.attention = PrototypeAttention(proto_dim, hidden_dim=proto_dim // 2)

        self.update_net = nn.Sequential(
            nn.Linear(proto_dim * 2, proto_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(proto_dim, proto_dim)
        )

        self.gate_net = nn.Sequential(
            nn.Linear(proto_dim, proto_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(proto_dim // 2, 1),
            nn.Sigmoid()
        )

        self.class_specific_decay = nn.Parameter(torch.ones(num_refinement_steps) * 0.5)

    def initial_prototype(self, support_embeddings, support_labels, n_way):
        """Compute initial prototype as class-wise mean."""
        prototypes = torch.zeros(n_way, support_embeddings.size(1), device=support_embeddings.device)

        for k in range(n_way):
            mask = (support_labels == k)
            if mask.sum() > 0:
                prototypes[k] = support_embeddings[mask].mean(0)

        return prototypes

    def refine_prototypes(self, prototypes, support_embeddings, support_labels,
                         query_embeddings, n_way, step):
        """
        Refine prototypes using query information.

        Args:
            prototypes: Current prototypes [n_way, proto_dim]
            support_embeddings: Support set embeddings
            support_labels: Support set labels
            query_embeddings: Query set embeddings
            n_way: Number of classes
            step: Current refinement step

        Returns:
            refined_prototypes: Updated prototypes
        """
        projected_prototypes = self.proto_projection(prototypes)
        projected_support = self.proto_projection(support_embeddings)
        projected_query = self.proto_projection(query_embeddings)

        attention_weights = self.attention(projected_support, projected_query)

        refined_prototypes = []

        for k in range(n_way):
            proto = projected_prototypes[k]

            mask = (support_labels == k)
            class_support = projected_support[mask]

            if class_support.size(0) > 0:
                class_attention = attention_weights[:, mask]

                query_aware_support = (class_attention.unsqueeze(-1) * class_support.unsqueeze(0)).sum(1)
                query_aware_support = query_aware_support / (class_attention.sum(1, keepdim=True) + 1e-8)

                combined = torch.cat([proto, query_aware_support.mean(0)], dim=-1)
                update = self.update_net(combined)

                gate = self.gate_net(proto).squeeze(-1)
                decay = self.class_specific_decay[step].clamp(0.3, 0.9)

                new_proto = decay * proto + (1 - decay) * update

                refined_prototypes.append(new_proto)
            else:
                refined_prototypes.append(proto)

        return torch.stack(refined_prototypes)


class DynamicPrototypeClassifier(nn.Module):
    """
    Dynamic Prototype Classifier with iterative refinement.

    Combines encoder, prototype refinement, and classification.
    """

    def __init__(self, in_channels=3, hidden_dim=256, proto_dim=128,
                 num_refinement_steps=3, dropout=0.3, n_way=5):
        super().__init__()

        self.encoder = ConvEncoder(in_channels, hidden_dim)
        self.refiner = DynamicPrototypeRefiner(
            self.encoder.out_features, proto_dim, num_refinement_steps, dropout
        )

        self.classifier = nn.Sequential(
            nn.Linear(proto_dim, proto_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(proto_dim, n_way)
        )

        self.n_way = n_way

    def forward(self, support_images, query_images, support_labels):
        """
        Forward pass for one episode.

        Args:
            support_images: Support set images [N*K, C, H, W]
            query_images: Query set images [N*Q, C, H, W]
            support_labels: Labels for support set [N*K]

        Returns:
            query_logits: Classification logits
            prototypes: Final prototypes after refinement
            all_prototypes: Prototypes at each refinement step (for analysis)
        """
        support_embeddings = self.encoder(support_images)
        query_embeddings = self.encoder(query_images)

        n_way = support_labels.unique().size(0)
        self.n_way = n_way

        prototypes = self.refiner.initial_prototype(support_embeddings, support_labels, n_way)

        all_prototypes = [prototypes]

        for step in range(self.refiner.num_steps):
            prototypes = self.refiner.refine_prototypes(
                prototypes, support_embeddings, support_labels,
                query_embeddings, n_way, step
            )
            all_prototypes.append(prototypes)

        distances = torch.cdist(query_embeddings, prototypes)
        logits = -distances

        return logits, prototypes, all_prototypes

    def get_embeddings(self, images):
        """Get embeddings for a set of images."""
        return self.encoder(images)


# =============================================================================
# EPISODIC SAMPLER
# =============================================================================

class EpisodicSampler:
    """Samples episodes for few-shot learning training."""

    def __init__(self, dataset, n_way=5, k_shot=5, n_query=15):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.class_indices = dataset.class_indices

    def sample_episode(self):
        """Sample a single episode."""
        available_classes = list(self.class_indices.keys())

        if len(available_classes) < self.n_way:
            selected_classes = available_classes
        else:
            selected_classes = random.sample(available_classes, self.n_way)

        support_images, support_labels = [], []
        query_images, query_labels = [], []

        for class_idx in selected_classes:
            class_samples = self.dataset.class_indices[class_idx]

            if len(class_samples) < self.k_shot + self.n_query:
                selected = random.sample(class_samples, len(class_samples))
            else:
                selected = random.sample(class_samples, self.k_shot + self.n_query)

            for idx in selected[:self.k_shot]:
                img, _ = self.dataset[idx]
                support_images.append(img)
                support_labels.append(class_idx)

            for idx in selected[self.k_shot:self.k_shot + self.n_query]:
                img, _ = self.dataset[idx]
                query_images.append(img)
                query_labels.append(class_idx)

        return (
            torch.stack(support_images), torch.tensor(support_labels, dtype=torch.long),
            torch.stack(query_images), torch.tensor(query_labels, dtype=torch.long),
            selected_classes
        )


# =============================================================================
# XAI IMPLEMENTATION
# =============================================================================

class XAIExplainer:
    """Explainable AI module for Dynamic Prototype FSL."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.saliency = Saliency(model)

    def get_saliency_map(self, images, target_class=None):
        """Compute gradient-based saliency maps."""
        self.model.eval()
        images = images.to(self.device).requires_grad_(True)

        if target_class is None:
            dummy_support = torch.zeros(5, *images.shape[1:], device=images.device)
            dummy_labels = torch.arange(5, device=images.device)
            with torch.no_grad():
                logits, _, _ = self.model(dummy_support, images[:len(images)], dummy_labels)
                target_class = logits[:len(images)].argmax(dim=1)

        saliency = self.saliency.attribute(images, target=target_class)
        return saliency.cpu().detach()

    def get_integrated_gradients(self, images, target_class=None, n_steps=50):
        """Compute Integrated Gradients attribution."""
        self.model.eval()
        images = images.to(self.device).requires_grad_(True)

        ig = IntegratedGradients(self.model)
        baseline = torch.zeros_like(images)

        if target_class is None:
            dummy_support = torch.zeros(5, *images.shape[1:], device=images.device)
            dummy_labels = torch.arange(5, device=images.device)
            with torch.no_grad():
                logits, _, _ = self.model(dummy_support, images[:len(images)], dummy_labels)
                target_class = logits[:len(images)].argmax(dim=1)

        attributions = ig.attribute(images, baseline=baseline, target=target_class, n_steps=n_steps)
        return attributions.cpu().detach()

    def generate_gradcam(self, images, target_layer=None, target_class=None):
        """Generate Grad-CAM visualization."""
        self.model.eval()
        images = images.to(self.device)

        if target_layer is None:
            target_layer = self.model.encoder.conv_blocks[-2]

        gradients, activations = [], []

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        def forward_hook(module, input, output):
            activations.append(output)

        hooks = []
        hooks.append(target_layer.register_full_backward_hook(backward_hook))
        hooks.append(target_layer.register_forward_hook(forward_hook))

        images = images.clone().detach().requires_grad_(True)
        features = self.model.encoder.conv_blocks(images)

        dummy_support = torch.zeros(5, *images.shape[1:], device=images.device)
        dummy_labels = torch.arange(5, device=images.device)
        logits = self.model.classifier(features.flatten(1)[:1])

        if target_class is None:
            target_class = logits.argmax(dim=1)

        self.model.zero_grad()
        logits[0, target_class].backward()

        grad = gradients[0].cpu().detach()
        activation = activations[0].cpu().detach()

        for hook in hooks:
            hook.remove()

        weights = grad.mean(dim=(2, 3), keepdim=True)
        gradcam = (weights * activation).relu().squeeze().mean(dim=0)

        gradcam = F.interpolate(
            gradcam.unsqueeze(0).unsqueeze(0),
            size=(images.shape[2], images.shape[3]),
            mode='bilinear', align_corners=False
        ).squeeze()

        gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)

        return gradcam.numpy()

    def visualize_explanation(self, image, saliency, save_path=None):
        """Visualize saliency map with overlay."""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        img_np = image.cpu().numpy().transpose(1, 2, 0)
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        img_np = np.clip(img_np * std + mean, 0, 1)

        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        saliency_np = saliency.squeeze().numpy()
        if saliency_np.ndim > 2:
            saliency_np = saliency_np.mean(axis=0)
        saliency_np = (saliency_np - saliency_np.min()) / (saliency_np.max() - saliency_np.min() + 1e-8)

        axes[1].imshow(saliency_np, cmap='jet')
        axes[1].set_title('Saliency Map')
        axes[1].axis('off')

        axes[2].imshow(img_np)
        axes[2].imshow(saliency_np, cmap='jet', alpha=0.6)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

        return fig

    def visualize_prototype_evolution(self, all_prototypes, support_labels, n_way, save_path=None):
        """Visualize prototype evolution during refinement."""
        fig, axes = plt.subplots(1, len(all_prototypes), figsize=(4 * len(all_prototypes), 4))

        for step, prototypes in enumerate(all_prototypes):
            proto_np = prototypes.cpu().numpy()
            proto_2d = proto_np[:, :2] if proto_np.shape[1] >= 2 else proto_np

            for k in range(n_way):
                axes[step].scatter(proto_2d[k, 0], proto_2d[k, 1], s=100, label=f'Class {k}')

            axes[step].set_title(f'Prototype Step {step}')
            axes[step].set_xlabel('Dim 1')
            axes[step].set_ylabel('Dim 2')
            if step == 0:
                axes[step].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

        return fig


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

class MetricsCalculator:
    """Computes evaluation metrics for Dynamic Prototype FSL."""

    def __init__(self, n_bins=15):
        self.n_bins = n_bins

    def compute_all_metrics(self, y_true, y_pred, y_prob=None, attributions=None):
        """Compute all evaluation metrics."""
        metrics = {}

        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)

        if y_prob is not None:
            metrics['ece'] = self._compute_ece(y_true, y_pred, y_prob)

        if attributions is not None:
            metrics['attribution_sparsity'] = self._compute_sparsity(attributions)

        return metrics

    def _compute_ece(self, y_true, y_pred, y_prob, n_bins=15):
        """Compute Expected Calibration Error."""
        confidences = np.max(y_prob, axis=1)
        accuracies = (y_pred == y_true).astype(float)

        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            bin_mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if bin_mask.sum() > 0:
                bin_acc = accuracies[bin_mask].mean()
                bin_conf = confidences[bin_mask].mean()
                ece += (bin_mask.sum() / len(y_true)) * abs(bin_acc - bin_conf)

        return ece

    def _compute_sparsity(self, attributions):
        """Compute attribution sparsity."""
        if attributions is None:
            return None

        attributions = np.abs(attributions)
        threshold = attributions.max() * 0.01
        non_zero = (attributions > threshold).sum()
        sparsity = 1 - (non_zero / attributions.size)

        return sparsity

    def compute_per_class_metrics(self, y_true, y_pred, n_classes):
        """Compute per-class precision, recall, F1."""
        precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

        return {
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist()
        }

    def statistical_test(self, scores1, scores2, method='ttest'):
        """Perform statistical significance test."""
        if method == 'ttest':
            stat, p_value = ttest_ind(scores1, scores2)
        else:
            stat, p_value = wilcoxon(scores1, scores2)

        return {'statistic': stat, 'p_value': p_value, 'significant': p_value < 0.05}


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

class DynamicProtoTrainer:
    """Training pipeline for Dynamic Prototype FSL."""

    def __init__(self, model, optimizer, scheduler=None, device='cuda'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

        self.metrics_calc = MetricsCalculator()

    def train_episode(self, support_images, support_labels, query_images, query_labels):
        """Train on a single episode."""
        self.model.train()

        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)

        self.optimizer.zero_grad()

        query_logits, _, _ = self.model(support_images, query_images, support_labels)

        loss = F.cross_entropy(query_logits, query_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        with torch.no_grad():
            preds = query_logits.argmax(dim=1)
            acc = (preds == query_labels).float().mean().item()

        return loss.item(), acc

    def evaluate(self, dataset, n_way=5, k_shot=5, n_query=15, n_episodes=100):
        """Evaluate the model on a dataset."""
        self.model.eval()

        all_preds, all_labels, all_probs = [], [], []

        sampler = EpisodicSampler(dataset, n_way, k_shot, n_query)

        with torch.no_grad():
            for _ in range(n_episodes):
                support_imgs, support_lbls, query_imgs, query_lbls, _ = sampler.sample_episode()

                support_imgs = support_imgs.to(self.device)
                support_lbls = support_lbls.to(self.device)
                query_imgs = query_imgs.to(self.device)

                query_logits, _, _ = self.model(support_imgs, query_imgs, support_lbls)

                probs = F.softmax(query_logits, dim=1).cpu().numpy()
                preds = query_logits.argmax(dim=1).cpu().numpy()
                labels = query_lbls.numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)
                all_probs.extend(probs)

        return self.metrics_calc.compute_all_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs)
        )

    def train_full(self, train_dataset, val_dataset, n_episodes=1500,
                   n_way=5, k_shot=5, n_query=15, save_dir='./checkpoints'):
        """Full training loop with episodic training."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        best_val_acc = 0.0
        patience_counter = 0

        pbar = tqdm(range(n_episodes), desc='Training')

        for episode in pbar:
            sampler = EpisodicSampler(train_dataset, n_way, k_shot, n_query)
            support_imgs, support_lbls, query_imgs, query_lbls, _ = sampler.sample_episode()

            loss, acc = self.train_episode(support_imgs, support_lbls, query_imgs, query_lbls)

            self.train_losses.append(loss)
            self.train_accs.append(acc)

            if episode % 50 == 0:
                val_metrics = self.evaluate(val_dataset, n_episodes=50, n_way=n_way, k_shot=k_shot, n_query=n_query)
                val_acc = val_metrics['accuracy']
                self.val_losses.append(val_metrics.get('loss', loss))
                self.val_accs.append(val_acc)

                pbar.set_postfix({'loss': f'{loss:.4f}', 'acc': f'{acc:.4f}', 'val_acc': f'{val_acc:.4f}'})

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(self.model.state_dict(), save_dir / 'best_model.pth')
                    patience_counter = 0
                else:
                    patience_counter += 1

                if self.scheduler:
                    self.scheduler.step()

                if patience_counter >= CONFIG['patience']:
                    print(f"\nEarly stopping at episode {episode}")
                    break

        return self.train_losses, self.train_accs


# =============================================================================
# VISUALIZATION
# =============================================================================

class Visualizer:
    """Visualization utilities for Dynamic Prototype FSL."""

    def __init__(self, output_dir='./visualizations', class_names=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names or [f'Class_{i}' for i in range(8)]
        self.xai_explainer = None

    def plot_training_history(self, train_losses, train_accs, val_losses, val_accs):
        """Plot training and validation curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        episodes = range(len(train_losses))
        val_episodes = range(0, len(train_losses), 50)

        axes[0].plot(episodes, train_losses, 'b-', alpha=0.5, label='Training Loss')
        if len(val_losses) == len(val_episodes):
            axes[0].plot(list(val_episodes), val_losses, 'r-', linewidth=2, label='Validation Loss')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(episodes, train_accs, 'b-', alpha=0.5, label='Training Accuracy')
        if len(val_accs) == len(val_episodes):
            axes[1].plot(list(val_episodes), val_accs, 'r-', linewidth=2, label='Validation Accuracy')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

        return fig

    def plot_confusion_matrix(self, y_true, y_pred, class_names=None):
        """Plot normalized confusion matrix."""
        if class_names is None:
            class_names = self.class_names[:len(np.unique(y_true)) + 1]

        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax,
                   cbar_kws={'label': 'Proportion'})

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Normalized Confusion Matrix - Dynamic Prototype FSL')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

        return fig

    def plot_per_class_metrics(self, metrics, class_names=None):
        """Plot per-class precision, recall, F1 scores."""
        if class_names is None:
            class_names = self.class_names

        n_classes = len(class_names)
        precision = metrics.get('precision_per_class', [0] * n_classes)
        recall = metrics.get('recall_per_class', [0] * n_classes)
        f1 = metrics.get('f1_per_class', [0] * n_classes)

        x = np.arange(n_classes)
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.bar(x - width, precision[:n_classes], width, label='Precision', color='#2ecc71')
        ax.bar(x, recall[:n_classes], width, label='Recall', color='#3498db')
        ax.bar(x + width, f1[:n_classes], width, label='F1-Score', color='#e74c3c')

        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names[:n_classes], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.1])

        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_class_metrics.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

        return fig

    def plot_calibration_curve(self, y_true, y_prob, n_bins=10):
        """Plot reliability diagram for calibration."""
        conf_true = np.max(y_prob, axis=1)
        conf_pred = y_true == y_prob.argmax(axis=1)

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

        prob_true, prob_pred = calibration_curve(conf_pred, conf_true, n_bins=n_bins)

        ax.plot(prob_pred, prob_true, 'o-', color='#3498db', linewidth=2, markersize=8, label='Dynamic Proto FSL')

        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curve (Reliability Diagram)')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'calibration_curve.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

        return fig

    def plot_learning_curves(self, train_history, val_history=None):
        """Plot learning curves."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for name, history in train_history.items():
            ax.plot(history, label=name, linewidth=2)

        ax.set_xlabel('Episode / Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'learning_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

        return fig


# =============================================================================
# SYNTHETIC DATA GENERATOR
# =============================================================================

def generate_synthetic_dataset(output_dir='./data', num_classes=8, images_per_class=160,
                               image_size=84, seed=42):
    """Generate synthetic dataset for testing."""
    import shutil

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(output_dir)
    (output_dir / 'train').mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic dataset: {num_classes} classes, {images_per_class} images/class")

    for class_idx in range(num_classes):
        class_name = f'class_{class_idx:02d}'
        class_dir = output_dir / 'train' / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        base_hue = (class_idx * 360 // num_classes) / 360.0

        for img_idx in range(images_per_class):
            img = np.zeros((image_size, image_size, 3), dtype=np.uint8)

            pattern_type = class_idx % 4

            if pattern_type == 0:
                color_h = (base_hue + np.random.uniform(-0.05, 0.05)) % 1.0
                color = np.array([int(255 * color_h), 150 + np.random.uniform(-50, 50), 150 + np.random.uniform(-50, 50)])
                cv2.rectangle(img, (10, 10), (image_size-10, image_size-10), color.tolist(), -1)

            elif pattern_type == 1:
                center = (image_size // 2, image_size // 2)
                radius = image_size // 3
                color_h = (base_hue + 0.25) % 1.0
                color = np.array([int(255 * color_h), 200, 200])
                cv2.circle(img, center, radius, color.tolist(), -1)

            elif pattern_type == 2:
                for i in range(5):
                    x = np.random.randint(10, image_size - 10)
                    y = np.random.randint(10, image_size - 10)
                    size = np.random.randint(10, 20)
                    color_h = (base_hue + i * 0.1) % 1.0
                    color = np.array([int(255 * color_h), 180, 180])
                    cv2.circle(img, (x, y), size, color.tolist(), -1)

            else:
                pts = [[np.random.randint(5, image_size - 5), np.random.randint(5, image_size - 5)] for _ in range(6)]
                pts = np.array(pts, dtype=np.int32)
                color_h = (base_hue + 0.5) % 1.0
                color = np.array([int(255 * color_h), 160, 160])
                cv2.fillPoly(img, [pts], color.tolist())

            noise = np.random.randint(-20, 20, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            img_path = class_dir / f'img_{img_idx:04d}.jpg'
            cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print(f"  Generated {class_name}: {images_per_class} images")

    print(f"Synthetic dataset saved to: {output_dir}")
    return output_dir


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_experiment(data_root='./data', output_dir='./dynamic_proto_output', config=CONFIG):
    """Run complete Dynamic Prototype FSL experiment pipeline."""
    print("=" * 60)
    print("Dynamic Prototype FSL: Few-Shot Learning with Dynamic Refinement")
    print("=" * 60)
    print(f"Device: {config['device']}")
    print(f"Output Directory: {output_dir}")
    print()

    output_dir = Path(output_dir)
    visualizer = Visualizer(output_dir / 'visualizations')

    print("Step 1: Loading datasets...")
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])

    train_dataset = FSLDataset(data_root, split='train', transform=transform)
    val_dataset = FSLDataset(data_root, split='val')
    test_dataset = FSLDataset(data_root, split='test')

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    print()

    print("Step 2: Initializing model...")
    model = DynamicPrototypeClassifier(
        in_channels=3,
        hidden_dim=config['hidden_dim'],
        proto_dim=config['proto_dim'],
        num_refinement_steps=config['refinement_steps'],
        dropout=config['dropout'],
        n_way=config['n_way']
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    print("Step 3: Setting up training...")
    optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['episodes'], eta_min=1e-6)

    trainer = DynamicProtoTrainer(model, optimizer, scheduler, device=config['device'])

    print("Step 4: Training model...")
    trainer.train_full(
        train_dataset, val_dataset,
        n_episodes=config['episodes'],
        n_way=config['n_way'],
        k_shot=config['k_shot'],
        n_query=config['n_query'],
        save_dir=output_dir / 'checkpoints'
    )
    print()

    print("Step 5: Evaluating model...")

    test_metrics = trainer.evaluate(
        test_dataset,
        n_episodes=200,
        n_way=config['n_way'],
        k_shot=config['k_shot'],
        n_query=config['n_query']
    )

    print("\nTest Set Metrics:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"  F1 (Micro): {test_metrics['f1_micro']:.4f}")
    print(f"  F1 (Weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    if 'ece' in test_metrics:
        print(f"  ECE: {test_metrics['ece']:.4f}")
    if 'attribution_sparsity' in test_metrics:
        print(f"  Attribution Sparsity: {test_metrics['attribution_sparsity']:.4f}")
    print()

    print("Step 6: Generating visualizations...")

    visualizer.plot_training_history(
        trainer.train_losses, trainer.train_accs,
        trainer.val_losses, trainer.val_accs
    )

    sampler = EpisodicSampler(test_dataset, config['n_way'], config['k_shot'], config['n_query'])
    support_imgs, support_lbls, query_imgs, query_lbls, selected_classes = sampler.sample_episode()

    all_preds, all_labels, all_probs = [], [], []

    for _ in range(10):
        s_imgs, s_lbls, q_imgs, q_lbls, _ = sampler.sample_episode()
        with torch.no_grad():
            q_logits, _, _ = model(
                s_imgs.to(config['device']),
                q_imgs.to(config['device']),
                s_lbls.to(config['device'])
            )
        all_preds.extend(q_logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(q_lbls.numpy())
        all_probs.extend(F.softmax(q_logits, dim=1).cpu().numpy())

    class_names = [train_dataset.idx_to_class[i] for i in sorted(selected_classes)]
    visualizer.plot_confusion_matrix(np.array(all_labels), np.array(all_preds), class_names)

    print("Step 7: Generating XAI visualizations...")

    xai = XAIExplainer(model, config['device'])
    visualizer.xai_explainer = xai

    sample_images = query_imgs[:5]

    for i, img in enumerate(sample_images):
        saliency = xai.get_saliency_map(img.unsqueeze(0))
        xai.visualize_explanation(img, saliency[0], save_path=visualizer.output_dir / f'xai_saliency_{i}.png')

    print("\n" + "=" * 60)
    print("Dynamic Prototype FSL Experiment Complete!")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")

    results = {
        'config': config,
        'test_metrics': test_metrics,
        'total_params': total_params,
        'trainable_params': trainable_params
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return model, results


if __name__ == '__main__':
    data_root = CONFIG['data_root']
    output_dir = CONFIG['output_dir']

    run_experiment(data_root, output_dir, CONFIG)