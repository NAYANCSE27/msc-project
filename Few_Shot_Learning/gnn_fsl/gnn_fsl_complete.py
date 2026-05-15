"""
GNN-FSL: Graph Neural Network for Few-Shot Learning with XAI Integration
========================================================================

This module implements a research-oriented GNN-based Few-Shot Learning model
integrated with Explainable AI (XAI) techniques for interpretable image classification.

Key Innovation: Unlike traditional FSL methods that use fixed distance metrics,
GNN-FSL learns to propagate information through a graph structure where nodes
represent samples and edges represent relationships, enabling learned comparison.

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
from sklearn.metrics import auc as sklearn_auc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

# For GNN operations
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, dense_to_sparse

# Gradient-based XAI
import cv2
from captum.attr import Saliency, IntegratedGradutions, GradientShap
from captum.concept import TCAV

# Configuration
CONFIG = {
    'seed': 42,
    'data_root': './data',
    'output_dir': './gnn_fsl_output',
    'num_classes': 8,
    'images_per_class': 160,
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'n_way': 5,
    'k_shot': 5,
    'n_query': 15,
    'episodes': 1000,
    'hidden_dim': 128,
    'gnn_layers': 3,
    'num_heads': 4,
    'dropout': 0.3,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'batch_size': 16,
    'epochs': 100,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'patience': 20,
}

def set_seed(seed):
    """Set all random seeds for reproducibility."""
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
GNN-FSL MATHEMATICAL FRAMEWORK
==============================

1. FEATURE ENCODING
   Given an image x_i, we extract features using a CNN encoder f_θ:

   h_i^(0) = f_θ(x_i) ∈ ℝ^d  (initial node features)

2. GRAPH CONSTRUCTION
   We construct an episode graph G = (V, E) where:
   - V = {v_1, v_2, ..., v_{N_support + N_query}} (all samples in episode)
   - E represents relationships between samples

   Adjacency matrix A is computed based on feature similarity:
   A_ij = exp(-||h_i - h_j||² / τ)  (thermal kernel)

   Or using k-nearest neighbors for sparse graphs.

3. GRAPH NEURAL NETWORK PROPAGATION
   We use Graph Attention Networks (GAT) for message passing:

   α_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j])) / √(d_k)

   h_i^(l+1) = σ(Σ_j∈N(i) α_ij W h_j)

   Multi-head attention with H heads:
   h_i^(l+1) = ||_{k=1}^{H} σ(Σ_j α_ij^k W^k h_j) / H

4. PROTOTYPE COMPUTATION IN GRAPH SPACE
   Class prototypes are computed from support set embeddings after GNN:

   c_k = (1/|S_k|) Σ_{x_i∈S_k} h_i^(L)

   where S_k is the support set for class k, L is the number of GNN layers.

5. CLASSIFICATION (ATTENTION-BASED PROTOTYPE MATCHING)
   Query classification uses attention-weighted prototype matching:

   α_k = softmax_k(-d(h_q^(L), c_k))  (attention weights)

   y_hat = argmax_k α_k

6. LOSS FUNCTIONS
   Episode Loss (N-way K-shot):
   L_episode = -Σ_{q∈Q} log P(y_q | x_q, S)

   Where P(y_q=k | x_q, S) = softmax(-d(h_q, c_k))

   Graph Regularization Loss:
   L_graph = λ Σ_{(i,j)∈E} ||h_i - h_j||² (encourages similar neighbors)

7. EVALUATION METRICS
   Accuracy: Acc = (TP + TN) / (TP + TN + FP + FN)

   F1-Score: F1 = 2 × (Precision × Recall) / (Precision + Recall)

   Expected Calibration Error (ECE):
   ECE = Σ_{b=1}^{B} |B_b|/n × |acc(B_b) - conf(B_b)|

   Attribution Sparsity:
   Sparsity = 1 - (non_zero_attributions / total_attributions)

"""

# =============================================================================
# DATASET IMPLEMENTATION
# =============================================================================

class FSLDataset(Dataset):
    """
    Few-Shot Learning Dataset with Graph-Aware Episodic Sampling.

    This dataset handles the stratified split and provides episodic
    sampling for few-shot learning training.
    """

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
            image = F.interpolate(image.unsqueeze(0), size=(84, 84), mode='bilinear', align_corners=False).squeeze(0)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_samples(self, class_idx, num_samples=None):
        """Get samples from a specific class."""
        indices = self.class_indices[class_idx]
        if num_samples is not None:
            indices = random.sample(indices, min(num_samples, len(indices)))
        return [self.samples[i] for i in indices]


def create_stratified_split(root_dir, output_dir, config=CONFIG):
    """
    Create stratified train/val/test split of the dataset.

    Ensures 80/10/10 distribution while maintaining class balance.
    """
    output_dir = Path(output_dir)
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in Path(root_dir).iterdir() if d.is_dir()])

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpeg'))
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
                    import shutil
                    shutil.copy(img_path, dest_path)

    print(f"Dataset split created in {output_dir}")
    return output_dir


# =============================================================================
# GRAPH CONSTRUCTION FOR GNN-FSL
# =============================================================================

class GraphConstructor:
    """
    Constructs graphs for Few-Shot Learning episodes.

    Each episode consists of:
    - Support set S: labeled samples for each of N classes
    - Query set Q: unlabeled samples to classify

    The graph connects all samples, with edges weighted by feature similarity.
    """

    def __init__(self, k_neighbors=5, temperature=0.1):
        self.k_neighbors = k_neighbors
        self.temperature = temperature

    def build_episode_graph(self, features, labels=None, is_support=None):
        """
        Build a graph for an episode.

        Args:
            features: Tensor of shape [num_samples, feature_dim]
            labels: Optional labels for support samples
            is_support: Boolean tensor indicating support samples

        Returns:
            edge_index: Graph connectivity
            edge_weight: Edge weights based on similarity
        """
        n_samples = features.shape[0]

        if n_samples <= self.k_neighbors:
            edge_index = torch.combinations(torch.arange(n_samples), 2)
            edge_index = torch.cat([edge_index, edge_index[:, [1, 0]]], dim=0).t()
            edge_weight = torch.ones(edge_index.shape[1])
            return edge_index, edge_weight

        distances = torch.cdist(features, features, p=2)

        _, nearest_indices = torch.topk(distances, k=min(self.k_neighbors + 1, n_samples), largest=False)

        row_indices = []
        col_indices = []
        for i in range(n_samples):
            neighbors = nearest_indices[i, 1:]
            for j in neighbors[:self.k_neighbors]:
                row_indices.extend([i, j.item()])
                col_indices.extend([j.item(), i])

        edge_index = torch.tensor([row_indices, col_indices], dtype=torch.long)

        edge_index, _ = add_self_loops(edge_index, num_nodes=n_samples)

        src, dst = edge_index[0], edge_index[1]
        similarities = torch.exp(-distances[src, dst] / self.temperature)
        edge_weight = similarities.float()

        return edge_index, edge_weight

    def build_dense_episode_graph(self, features, labels=None):
        """
        Build a fully connected graph for the episode.

        Used for smaller episodes where we want full connectivity.
        """
        n_samples = features.shape[0]

        edge_index = torch.combinations(torch.arange(n_samples), 2)
        edge_index = torch.cat([edge_index, edge_index[:, [1, 0]]], dim=0).t()
        edge_index, _ = add_self_loops(edge_index, num_nodes=n_samples)

        src, dst = edge_index[0], edge_index[1]
        distances = torch.cdist(features, features, p=2)
        similarities = torch.exp(-distances[src, dst] / self.temperature)
        edge_weight = similarities.float()

        return edge_index, edge_weight


# =============================================================================
# MODEL ARCHITECTURE: GNN-FSL
# =============================================================================

class FeatureEncoder(nn.Module):
    """
    CNN Feature Encoder for GNN-FSL.

    Extracts high-level features from input images that will serve as
    initial node features in the graph.

    Architecture:
    - 4 convolutional blocks with batch normalization and ReLU
    - Adaptive average pooling to fixed size (4x4)
    - Output: 64-dimensional feature vector per image
    """

    def __init__(self, in_channels=3, hidden_dim=64):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            self._make_conv_block(in_channels, 32, kernel_size=3, padding=1),
            self._make_conv_block(32, 64, kernel_size=3, padding=1),
            self._make_conv_block(64, 128, kernel_size=3, padding=1),
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
        features = self.conv_blocks(x)
        return features.view(features.size(0), -1)


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for GNN-FSL.

    Implements multi-head attention mechanism for graph-structured data.

    Attention computation:
    α_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j])) / √(d_k)

    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, in_features, out_features, n_heads=4, dropout=0.3):
        super().__init__()

        self.n_heads = n_heads
        self.out_features = out_features // n_heads

        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * self.out_features, 1)
        self.dropout = nn.Dropout(dropout)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Graph connectivity [2, num_edges]
        """
        h = self.W(x).view(-1, self.n_heads, self.out_features)

        src, dst = edge_index
        h_src = h[src]
        h_dst = h[dst]

        h_concat = torch.cat([h_src, h_dst], dim=-1)
        e = self.leaky_relu(self.a(h_concat).squeeze(-1))

        attention_scores = torch.zeros(x.size(0), x.size(0), device=x.device)
        attention_scores[src, dst] = e

        attention_scores = attention_scores.masked_fill(
            torch.eye(x.size(0), device=x.device).bool(), float('-inf')
        )

        alpha = F.softmax(attention_scores, dim=1)
        alpha = self.dropout(alpha)

        output = torch.matmul(alpha, h.view(-1, self.n_heads * self.out_features))

        return output


class GNNBlock(nn.Module):
    """
    Single GNN Block with Graph Attention.

    Each block applies:
    1. Multi-head self-attention
    2. Residual connection and layer normalization
    3. Feed-forward network (optional)
    """

    def __init__(self, in_features, out_features, n_heads=4, dropout=0.3):
        super().__init__()

        self.attention = GraphAttentionLayer(in_features, out_features, n_heads, dropout)
        self.norm1 = nn.LayerNorm(out_features)
        self.norm2 = nn.LayerNorm(out_features)

        self.ff = nn.Sequential(
            nn.Linear(out_features, out_features * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_features * 2, out_features)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        h_attn = self.attention(x, edge_index)
        h = self.norm1(x + self.dropout(h_attn))

        h_ff = self.ff(h)
        h = self.norm2(h + self.dropout(h_ff))

        return h


class GNNFSLModel(nn.Module):
    """
    GNN-Based Few-Shot Learning Model.

    Architecture:
    1. Feature Encoder: CNN to extract initial node features
    2. GNN Layers: Graph attention for feature propagation
    3. Prototype Computation: Class prototypes from support embeddings
    4. Classification: Attention-weighted prototype matching

    The model learns both:
    - Good feature representations for images
    - How to propagate information through the graph structure
    """

    def __init__(self, in_channels=3, hidden_dim=128, gnn_layers=3,
                 n_heads=4, dropout=0.3, num_classes=5):
        super().__init__()

        self.encoder = FeatureEncoder(in_channels, hidden_dim // 2)
        encoder_out_dim = self.encoder.out_features

        self.node_embedding = nn.Linear(encoder_out_dim, hidden_dim)

        self.gnn_blocks = nn.ModuleList([
            GNNBlock(hidden_dim, hidden_dim, n_heads, dropout)
            for _ in range(gnn_layers)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.graph_constructor = GraphConstructor(k_neighbors=5, temperature=0.1)

    def forward(self, support_images, query_images, support_labels, n_way):
        """
        Forward pass for one episode.

        Args:
            support_images: Support set images [N*K, C, H, W]
            query_images: Query set images [N*Q, C, H, W]
            support_labels: Labels for support set [N*K]
            n_way: Number of classes in episode

        Returns:
            query_logits: Classification logits for query set
            all_embeddings: All embeddings after GNN
            prototypes: Class prototypes
        """
        all_images = torch.cat([support_images, query_images], dim=0)
        n_support = support_images.size(0)
        n_total = all_images.size(0)

        features = self.encoder(all_images)
        node_features = self.node_embedding(features)

        edge_index, edge_weight = self.graph_constructor.build_dense_episode_graph(node_features)

        for gnn_block in self.gnn_blocks:
            node_features = gnn_block(node_features, edge_index)

        support_embeddings = node_features[:n_support]
        query_embeddings = node_features[n_support:]

        prototypes = self._compute_prototypes(support_embeddings, support_labels, n_way)

        query_logits = self._compute_logits(query_embeddings, prototypes)

        return query_logits, node_features, prototypes

    def _compute_prototypes(self, support_embeddings, support_labels, n_way):
        """Compute class prototypes from support set embeddings."""
        prototypes = torch.zeros(n_way, support_embeddings.size(1), device=support_embeddings.device)

        for k in range(n_way):
            mask = (support_labels == k)
            if mask.sum() > 0:
                prototypes[k] = support_embeddings[mask].mean(0)

        return prototypes

    def _compute_logits(self, embeddings, prototypes):
        """Compute classification logits using negative distance to prototypes."""
        distances = torch.cdist(embeddings, prototypes)
        logits = -distances
        return logits

    def get_episode_embeddings(self, images):
        """Get GNN embeddings for a set of images (used for XAI)."""
        features = self.encoder(images)
        node_features = self.node_embedding(features)

        edge_index, edge_weight = self.graph_constructor.build_dense_episode_graph(node_features)

        for gnn_block in self.gnn_blocks:
            node_features = gnn_block(node_features, edge_index)

        return node_features


class PrototypeAttentionLayer(nn.Module):
    """
    Attention-weighted prototype computation.

    Learns to weight support samples differently when computing prototypes,
    allowing the model to focus on more informative samples.
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.attention_weights = nn.Linear(feature_dim, 1)

    def forward(self, support_embeddings, support_labels, n_way):
        """
        Compute attention-weighted prototypes.

        a_i = softmax(v^T tanh(W h_i))
        c_k = Σ_i a_i * h_i / Σ_i a_i  (for samples of class k)
        """
        attention_scores = self.attention_weights(support_embeddings)
        attention_scores = attention_scores.squeeze(-1)

        prototypes = torch.zeros(n_way, support_embeddings.size(1), device=support_embeddings.device)

        for k in range(n_way):
            mask = (support_labels == k)
            if mask.sum() > 0:
                class_embeddings = support_embeddings[mask]
                class_attention = attention_scores[mask]

                attention_weights = F.softmax(class_attention, dim=0)
                weighted_sum = (class_embeddings * attention_weights.unsqueeze(-1)).sum(0)

                prototypes[k] = weighted_sum

        return prototypes


# =============================================================================
# EPISODIC SAMPLER
# =============================================================================

class EpisodicSampler:
    """
    Samples episodes for few-shot learning training.

    Each episode contains:
    - n_way classes
    - k_shot support samples per class
    - n_query query samples per class
    """

    def __init__(self, dataset, n_way=5, k_shot=5, n_query=15):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query

        self.class_indices = dataset.class_indices

    def sample_episode(self):
        """
        Sample a single episode.

        Returns:
            support_images, support_labels: Support set
            query_images, query_labels: Query set
            selected_classes: The classes selected for this episode
        """
        available_classes = list(self.class_indices.keys())

        if len(available_classes) < self.n_way:
            selected_classes = available_classes
        else:
            selected_classes = random.sample(available_classes, self.n_way)

        support_images = []
        support_labels = []
        query_images = []
        query_labels = []

        for class_idx in selected_classes:
            class_samples = self.dataset.class_indices[class_idx]

            if len(class_samples) < self.k_shot + self.n_query:
                selected = random.sample(class_samples, len(class_samples))
            else:
                selected = random.sample(class_samples, self.k_shot + self.n_query)

            support_indices = selected[:self.k_shot]
            query_indices = selected[self.k_shot:self.k_shot + self.n_query]

            for idx in support_indices:
                img, label = self.dataset[idx]
                support_images.append(img)
                support_labels.append(class_idx)

            for idx in query_indices:
                img, label = self.dataset[idx]
                query_images.append(img)
                query_labels.append(class_idx)

        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels, dtype=torch.long)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels, dtype=torch.long)

        return (support_images, support_labels), (query_images, query_labels), selected_classes


# =============================================================================
# XAI IMPLEMENTATION
# =============================================================================

class XAIExplainer:
    """
    Explainable AI module for GNN-FSL.

    Provides multiple XAI techniques:
    1. Gradient-based: Saliency Maps, Integrated Gradients
    2. Attention-based: GNN Attention visualization
    3. Prototype-based: Nearest support samples
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.saliency = Saliency(model)

    def get_saliency_map(self, images, target_class=None):
        """
        Compute saliency maps using gradient-based method.

        Saliency = ∂(output_class) / ∂(input)

        Args:
            images: Input images [B, C, H, W]
            target_class: Target class for attribution (None = predicted class)

        Returns:
            saliency_maps: Gradient saliency [B, H, W]
        """
        self.model.eval()
        images = images.to(self.device).requires_grad_(True)

        if target_class is None:
            with torch.no_grad():
                logits = self._get_logits(images)
                target_class = logits.argmax(dim=1)

        saliency = self.saliency.attribute(images, target=target_class)

        return saliency.cpu().detach()

    def get_integrated_gradients(self, images, target_class=None, n_steps=50):
        """
        Compute Integrated Gradients attribution.

        IG(x) = (x - x') × Σ_{i=1}^{n} ∂F(x' + α_i × (x - x'))/∂x × (1/n)

        where x' is a baseline (typically zero image).
        """
        self.model.eval()
        images = images.to(self.device).requires_grad_(True)

        ig = IntegratedGradients(self.model)

        baseline = torch.zeros_like(images)

        if target_class is None:
            with torch.no_grad():
                logits = self._get_logits(images)
                target_class = logits.argmax(dim=1)

        attributions = ig.attribute(
            images,
            baseline=baseline,
            target=target_class,
            n_steps=n_steps
        )

        return attributions.cpu().detach()

    def get_gnn_attention_weights(self, images):
        """
        Extract attention weights from GNN layers for visualization.

        Returns attention maps showing which nodes influenced each prediction.
        """
        self.model.eval()

        attention_weights = []

        def hook_fn(module, input, output):
            attention_weights.append(output)

        hooks = []
        for block in self.model.gnn_blocks:
            hook = block.attention.register_forward_hook(hook_fn)
            hooks.append(hook)

        with torch.no_grad():
            embeddings = self.model.get_episode_embeddings(images)

        for hook in hooks:
            hook.remove()

        return embeddings, attention_weights

    def generate_gradcam(self, images, target_layer=None, target_class=None):
        """
        Generate Grad-CAM style visualization for CNN features.

        Grad-CAM: α_k × ReLU(Σ_i A_i^c)

        where α_k = (1/Z) Σ_i ∂y^c/∂A_i^k
        """
        self.model.eval()
        images = images.to(self.device)

        if target_layer is None:
            target_layer = self.model.encoder.conv_blocks[-2]

        gradients = []
        activations = []

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        def forward_hook(module, input, output):
            activations.append(output)

        hooks = []
        hooks.append(target_layer.register_full_backward_hook(backward_hook))
        hooks.append(target_layer.register_forward_hook(forward_hook))

        images = images.clone().detach().requires_grad_(True)
        features = self.model.encoder.conv_blocks(images)
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
        gradcam = (weights * activation).relu()
        gradcam = gradcam.squeeze().mean(dim=0)

        gradcam = F.interpolate(
            gradcam.unsqueeze(0).unsqueeze(0),
            size=(images.shape[2], images.shape[3]),
            mode='bilinear',
            align_corners=False
        ).squeeze()

        gradcam = (gradcam - gradcam.min()) / (gradcam.max() - gradcam.min() + 1e-8)

        return gradcam.numpy()

    def visualize_explanation(self, image, saliency, save_path=None):
        """Visualize saliency map overlaid on original image."""
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        img_np = image.cpu().numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)

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
        else:
            plt.show()

        return fig


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

class MetricsCalculator:
    """
    Computes evaluation metrics for GNN-FSL model.

    Metrics include:
    - Accuracy, F1-Score, Precision, Recall
    - Expected Calibration Error (ECE)
    - Attribution Sparsity
    - Statistical significance tests
    """

    def __init__(self, n_bins=15):
        self.n_bins = n_bins

    def compute_all_metrics(self, y_true, y_pred, y_prob=None,
                           attributions=None, class_names=None):
        """
        Compute all evaluation metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (for calibration)
            attributions: XAI attributions (for sparsity)
            class_names: Names of classes

        Returns:
            Dictionary of computed metrics
        """
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
        """
        Compute Expected Calibration Error.

        ECE = Σ_b |B_b|/n × |acc(B_b) - conf(B_b)|

        where B_b is the set of samples in confidence bin b.
        """
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
        """
        Compute attribution sparsity.

        Sparsity = 1 - (non_zero_attributions / total_attributions)

        Uses threshold of 1% of max absolute value.
        """
        if attributions is None:
            return None

        attributions = np.abs(attributions)
        threshold = attributions.max() * 0.01

        non_zero = (attributions > threshold).sum()
        total = attributions.size

        sparsity = 1 - (non_zero / total)

        return sparsity

    def compute_confusion_matrix(self, y_true, y_pred, class_names=None):
        """Compute normalized confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        return cm, cm_normalized

    def statistical_test(self, scores1, scores2, method='ttest'):
        """
        Perform statistical significance test between two sets of scores.

        Args:
            scores1: First set of scores
            scores2: Second set of scores
            method: 'ttest' for t-test, 'wilcoxon' for Wilcoxon signed-rank

        Returns:
            Dictionary with test statistic and p-value
        """
        if method == 'ttest':
            stat, p_value = ttest_ind(scores1, scores2)
        else:
            stat, p_value = wilcoxon(scores1, scores2)

        return {
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

class GNNFSSLTrainer:
    """
    Training pipeline for GNN-FSL model.

    Handles episodic training, validation, and evaluation.
    """

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
        self.xai_explainer = XAIExplainer(model, device)

    def train_episode(self, support_images, support_labels,
                     query_images, query_labels, n_way):
        """
        Train on a single episode.

        Args:
            support_images: Support set images
            support_labels: Support set labels
            query_images: Query set images
            query_labels: Query set labels
            n_way: Number of classes

        Returns:
            loss: Episode loss
            acc: Query set accuracy
        """
        self.model.train()

        support_images = support_images.to(self.device)
        support_labels = support_labels.to(self.device)
        query_images = query_images.to(self.device)
        query_labels = query_labels.to(self.device)

        self.optimizer.zero_grad()

        query_logits, _, _ = self.model(
            support_images, query_images, support_labels, n_way
        )

        loss = F.cross_entropy(query_logits, query_labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        with torch.no_grad():
            preds = query_logits.argmax(dim=1)
            acc = (preds == query_labels).float().mean().item()

        return loss.item(), acc

    def evaluate(self, dataset, n_way=5, k_shot=5, n_query=15, n_episodes=100):
        """
        Evaluate the model on a dataset.

        Args:
            dataset: Evaluation dataset
            n_way: Number of classes per episode
            k_shot: Support samples per class
            n_query: Query samples per class
            n_episodes: Number of episodes to evaluate

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        sampler = EpisodicSampler(dataset, n_way, k_shot, n_query)

        with torch.no_grad():
            for _ in range(n_episodes):
                (support_imgs, support_lbls), (query_imgs, query_lbls), _ = sampler.sample_episode()

                support_imgs = support_imgs.to(self.device)
                support_lbls = support_lbls.to(self.device)
                query_imgs = query_imgs.to(self.device)

                query_logits, _, _ = self.model(
                    support_imgs, query_imgs, support_lbls, n_way
                )

                probs = F.softmax(query_logits, dim=1).cpu().numpy()
                preds = query_logits.argmax(dim=1).cpu().numpy()
                labels = query_lbls.numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)
                all_probs.extend(probs)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        metrics = self.metrics_calc.compute_all_metrics(
            all_labels, all_preds, all_probs
        )

        return metrics

    def train_full(self, train_dataset, val_dataset, n_episodes=1000,
                   n_way=5, k_shot=5, n_query=15, save_dir='./checkpoints'):
        """
        Full training loop with episodic training.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        best_val_acc = 0.0
        patience_counter = 0

        pbar = tqdm(range(n_episodes), desc='Training')

        for episode in pbar:
            sampler = EpisodicSampler(train_dataset, n_way, k_shot, n_query)
            (support_imgs, support_lbls), (query_imgs, query_lbls), _ = sampler.sample_episode()

            loss, acc = self.train_episode(
                support_imgs, support_lbls, query_imgs, query_lbls, n_way
            )

            self.train_losses.append(loss)
            self.train_accs.append(acc)

            if episode % 50 == 0:
                val_metrics = self.evaluate(val_dataset, n_episodes=50,
                                           n_way=n_way, k_shot=k_shot, n_query=n_query)
                val_acc = val_metrics['accuracy']
                self.val_losses.append(val_metrics.get('loss', loss))
                self.val_accs.append(val_acc)

                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'acc': f'{acc:.4f}',
                    'val_acc': f'{val_acc:.4f}'
                })

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
    """
    Visualization utilities for GNN-FSL results.

    Generates:
    - Training curves
    - Confusion matrices
    - Per-class metrics
    - XAI visualizations
    - Graph structure visualizations
    """

    def __init__(self, output_dir='./visualizations', class_names=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names or [f'Class_{i}' for i in range(8)]

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
        plt.close()

        return fig

    def plot_confusion_matrix(self, y_true, y_pred, class_names=None):
        """Plot normalized confusion matrix."""
        if class_names is None:
            class_names = self.class_names[:len(np.unique(y_true)) + 1]

        cm, cm_normalized = confusion_matrix(y_true, y_pred), None

        fig, ax = plt.subplots(figsize=(10, 8))

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax,
                   cbar_kws={'label': 'Proportion'})

        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Normalized Confusion Matrix - GNN-FSL')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
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
        plt.close()

        return fig

    def plot_calibration_curve(self, y_true, y_prob, n_bins=10):
        """Plot reliability diagram for calibration."""
        conf_true = np.max(y_prob, axis=1)
        conf_pred = y_true == y_prob.argmax(axis=1)

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')

        prob_true, prob_pred = calibration_curve(conf_pred, conf_true, n_bins=n_bins)

        ax.plot(prob_pred, prob_true, 'o-', color='#3498db', linewidth=2, markersize=8,
               label=f'GNN-FSL (ECE = {np.mean(np.abs(prob_true - prob_pred)):.3f})')

        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curve (Reliability Diagram)')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'calibration_curve.png', dpi=150, bbox_inches='tight')
        plt.close()

        return fig

    def plot_learning_curves(self, train_history, val_history=None):
        """Plot learning curves comparing different runs or models."""
        fig, ax = plt.subplots(figsize=(10, 6))

        for i, (name, history) in enumerate(train_history.items()):
            ax.plot(history, label=name, linewidth=2)

        ax.set_xlabel('Episode / Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'learning_curves.png', dpi=150, bbox_inches='tight')
        plt.close()

        return fig

    def visualize_graph_structure(self, embeddings, labels, edge_index=None,
                                  title='Episode Graph Structure'):
        """Visualize the graph structure in embedding space."""
        fig, ax = plt.subplots(figsize=(10, 10))

        embeddings_2d = embeddings[:, :2].cpu().numpy()
        labels_np = labels.cpu().numpy()

        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                           c=labels_np, cmap='tab10', s=100, alpha=0.7)

        plt.colorbar(scatter, ax=ax, label='Class')

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(title)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'graph_structure.png', dpi=150, bbox_inches='tight')
        plt.close()

        return fig


# =============================================================================
# SYNTHETIC DATA GENERATOR
# =============================================================================

def generate_synthetic_dataset(output_dir='./data', num_classes=8, images_per_class=160,
                               image_size=84, seed=42):
    """
    Generate synthetic dataset for testing GNN-FSL.

    Creates synthetic images with distinguishable patterns for each class.
    """
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
                color = np.array([
                    int(255 * color_h),
                    int(150 + np.random.uniform(-50, 50)),
                    int(150 + np.random.uniform(-50, 50))
                ])
                cv2.rectangle(img, (10, 10), (image_size-10, image_size-10), color.tolist(), -1)

            elif pattern_type == 1:
                center = (image_size // 2, image_size // 2)
                radius = image_size // 3
                color_h = (base_hue + 0.25) % 1.0
                color = np.array([
                    int(255 * color_h),
                    200, 200
                ])
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
                pts = []
                for _ in range(6):
                    pts.append([
                        np.random.randint(5, image_size - 5),
                        np.random.randint(5, image_size - 5)
                    ])
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

def run_experiment(data_root='./data', output_dir='./gnn_fsl_output', config=CONFIG):
    """
    Run complete GNN-FSL experiment pipeline.

    1. Load/Prepare data
    2. Create datasets
    3. Initialize model
    4. Train model
    5. Evaluate model
    6. Generate visualizations and XAI
    """
    print("=" * 60)
    print("GNN-FSL: Graph Neural Network for Few-Shot Learning")
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
    model = GNNFSLModel(
        in_channels=3,
        hidden_dim=config['hidden_dim'],
        gnn_layers=config['gnn_layers'],
        n_heads=config['num_heads'],
        dropout=config['dropout'],
        num_classes=config['n_way']
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    print("Step 3: Setting up training...")
    optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)

    trainer = GNNFSSLTrainer(model, optimizer, scheduler, device=config['device'])

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
    (support_imgs, support_lbls), (query_imgs, query_lbls), selected_classes = sampler.sample_episode()

    with torch.no_grad():
        model.eval()
        query_logits, embeddings, prototypes = model(
            support_imgs.to(config['device']),
            query_imgs.to(config['device']),
            support_lbls.to(config['device']),
            config['n_way']
        )
        preds = query_logits.argmax(dim=1).cpu().numpy()

    all_preds = []
    all_labels = []
    all_probs = []

    for _ in range(10):
        (s_imgs, s_lbls), (q_imgs, q_lbls), _ = sampler.sample_episode()
        with torch.no_grad():
            q_logits, _, _ = model(
                s_imgs.to(config['device']),
                q_imgs.to(config['device']),
                s_lbls.to(config['device']),
                config['n_way']
            )
        all_preds.extend(q_logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(q_lbls.numpy())
        all_probs.extend(F.softmax(q_logits, dim=1).cpu().numpy())

    class_names = [train_dataset.idx_to_class[i] for i in sorted(selected_classes)]
    visualizer.plot_confusion_matrix(np.array(all_labels), np.array(all_preds), class_names)

    print("Step 7: Generating XAI visualizations...")

    xai = XAIExplainer(model, config['device'])

    sample_images = query_imgs[:5]

    for i, img in enumerate(sample_images):
        saliency = xai.get_saliency_map(img.unsqueeze(0))
        visualizer.xai_explainer.visualize_explanation(
            img, saliency[0],
            save_path=visualizer.output_dir / f'xai_saliency_{i}.png'
        )

    print("\n" + "=" * 60)
    print("GNN-FSL Experiment Complete!")
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
