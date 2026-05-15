# Few-Shot Explainable Plant Pathology: Prototypical Networks with Pixel-Level Attribution for Cucumber Disease Classification

## Comprehensive Technical Blueprint and Manuscript Structure

---

This document presents a mathematically rigorous technical framework for few-shot explainable plant pathology, targeting Q1 journal publication in IEEE Access, Computers and Electronics in Agriculture, or Information Fusion.

**Dataset Specifications:**
- 8 classes (6 disease + 2 healthy): Anthracnose, Bacterial Wilt, Belly Rot, Downy Mildew, Pythium Fruit Rot, Gummy Stem Blight, Fresh Leaves, Fresh Cucumber
- 160 images per class, 1,280 total images
- Domain: Precision Agriculture / Plant Pathology

---

# 1. MATHEMATICAL FORMULATION & ARCHITECTURE

## 1.1 Episodic Training Framework

The few-shot learning paradigm frames classification as a meta-learning problem where models learn to learn from limited samples. For an $N$-way $K$-shot task, we construct episodic batches as follows:

**Support Set $S$:** A set of $N \times K$ labeled images from $N$ distinct classes, each class contributing $K$ examples:
$$S = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N \times K}, \quad y_i \in \{1, 2, \ldots, N\}$$

**Query Set $Q$:** A set of $N \times Q_{query}$ unlabeled images sampled from the same $N$ classes:
$$Q = \{\mathbf{x}_j\}_{j=1}^{N \times Q_{query}}$$

**Episode:** One training iteration comprises one support set and one query set, where classes are randomly sampled without replacement from the training pool.

## 1.2 Embedding Function

Let $f_\theta: \mathbb{R}^{H \times W \times C} \rightarrow \mathbb{R}^D$ denote the convolutional backbone encoder with learnable parameters $\theta$. For ResNet-18 or MobileNetV3:

$$\mathbf{z}_i = f_\theta(\mathbf{x}_i), \quad \mathbf{z}_i \in \mathbb{R}^D$$

Where:
- Input: $\mathbf{x}_i \in \mathbb{R}^{224 \times 224 \times 3}$ (RGB image)
- ResNet-18 output: $D = 512$ (avgpool layer)
- MobileNetV3-Small output: $D = 576$
- MobileNetV3-Large output: $D = 960$

The backbone consists of:
- ResNet-18: 4 residual blocks (conv2_x, conv3_x, conv4_x, conv5_x) with 512-channel output
- MobileNetV3: 5 inverted residual blocks with SE attention, final conv layer at 960 channels

## 1.3 Prototype Computation

For each class $k \in \{1, 2, \ldots, N\}$, the prototype $\mathbf{c}_k \in \mathbb{R}^D$ is computed as the mean of embedded support vectors:

$$\mathbf{c}_k = \frac{1}{|S_k|} \sum_{(\mathbf{x}_i, y_i) \in S_k} f_\theta(\mathbf{x}_i)$$

Where $S_k = \{(\mathbf{x}_i, y_i) \in S : y_i = k\}$ is the subset of support images belonging to class $k$, and $|S_k| = K$ by construction.

**Prototype aggregation:**
- 5-way 1-shot: $N$ prototypes, each from 1 sample
- 5-way 5-shot: $N$ prototypes, each from 5 samples (averaged)

## 1.4 Distance Metric and Classification

Using squared Euclidean distance:

$$d(\mathbf{z}, \mathbf{c}_k) = \|\mathbf{z} - \mathbf{c}_k\|_2^2 = (\mathbf{z} - \mathbf{c}_k)^\top (\mathbf{z} - \mathbf{c}_k)$$

The probability distribution over classes via softmax over negative distances:

$$P_\theta(y = k \mid \mathbf{x}) = \frac{\exp(-d(f_\theta(\mathbf{x}), \mathbf{c}_k))}{\sum_{j=1}^{N} \exp(-d(f_\theta(\mathbf{x}), \mathbf{c}_j))} = \frac{\exp\left(-\|f_\theta(\mathbf{x}) - \mathbf{c}_k\|_2^2\right)}{\sum_{j=1}^{N} \exp\left(-\|f_\theta(\mathbf{x}) - \mathbf{c}_j\|_2^2\right)}$$

## 1.5 Cross-Entropy Loss

For each query sample, the loss is computed as:

$$\mathcal{L}_{CE} = -\sum_{k=1}^{N} \mathbb{1}[y_{query} = k] \log P_\theta(y = k \mid \mathbf{x}_{query})$$

For an episode with $Q_{query}$ query samples per class:

$$\mathcal{L}_{episode} = \frac{1}{N \times Q_{query}} \sum_{j=1}^{N \times Q_{query}} \mathcal{L}_{CE}^{(j)}$$

**Full episodic loss:**
$$\mathcal{L}_{total} = \mathbb{E}_{\mathcal{E}}[\mathcal{L}_{episode}]$$

Where $\mathcal{E}$ denotes the distribution over randomly constructed episodes.

---

# 2. EXPLAINABLE AI (XAI) INTEGRATION LAYER

## 2.1 Post-Hoc Attribution: Grad-CAM for Prototypical Networks

### 2.1.1 Gradient Computation Architecture

The challenge in applying Grad-CAM to Prototypical Networks lies in the prototype-distance computation. We backpropagate from the prototype-distance logits:

$$\text{logit}_k = -d(\mathbf{z}_{query}, \mathbf{c}_k) = -\|\mathbf{z}_{query} - \mathbf{c}_k\|_2^2$$

For the predicted class $\hat{k}$:
$$L_{Grad-CAM} = \text{logit}_{\hat{k}}$$

**Gradient flow:**
1. Forward pass: Compute query embedding $\mathbf{z}_{query} = f_\theta(\mathbf{x}_{query})$
2. Compute distance to each prototype: $d_k = \|\mathbf{z}_{query} - \mathbf{c}_k\|_2^2$
3. Compute softmax probabilities: $P_k = \frac{\exp(-d_k)}{\sum_j \exp(-d_j)}$
4. Backward pass: $\frac{\partial L_{Grad-CAM}}{\partial \mathbf{z}_{query}}$
5. Continue backprop to final convolutional feature map $A^L \in \mathbb{R}^{H_L \times W_L \times C_L}$

### 2.1.2 Grad-CAM Weight Calculation

Importance weights $\alpha_k^c$ for feature map channel $c$:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial \text{logit}_k}{\partial A_{ij}^{L,c}}$$

Where $Z = H_L \times W_L$ is the spatial dimension of the final feature map.

### 2.1.3 Heatmap Generation

The Grad-CAM heatmap for class $k$:

$$\text{Grad-CAM}_k^{ij} = \text{ReLU}\left(\sum_c \alpha_k^c A^{L,c}_{ij}\right)$$

This yields spatial attention over the input image, highlighting regions most influential for the prototype-distance computation.

## 2.2 Alternative: Integrated Gradients

### 2.2.1 Path Integral Formulation

Integrated Gradients computes attributions by integrating gradients along a path from a baseline $\mathbf{x}_{baseline}$ (e.g., zero image or mean dataset image) to the input $\mathbf{x}$:

$$\text{IG}_k(\mathbf{x}) = (\mathbf{x} - \mathbf{x}_{baseline}) \odot \int_{\alpha=0}^{1} \frac{\partial F_k(\alpha \cdot \mathbf{x}_{baseline} + (1-\alpha) \cdot \mathbf{x})}{\partial \mathbf{x}} \, d\alpha$$

Where $F_k(\cdot)$ is the model's probability for class $k$, and $\odot$ denotes element-wise multiplication.

### 2.2.2 Discretization Approximation

In practice, with $R = 50$ steps:
$$\text{IG}_k(\mathbf{x}) \approx \frac{\mathbf{x} - \mathbf{x}_{baseline}}{R} \sum_{r=1}^{R} \nabla_{\mathbf{x}} F_k\left(\mathbf{x}_{baseline} + \frac{r}{R}(\mathbf{x} - \mathbf{x}_{baseline})\right)$$

## 2.3 Intrinsic Interpretability: ProtoPNet Extension

### 2.3.1 Prototype Layer Architecture

Instead of global prototype vectors, learn localized prototype patches:

$$\mathbf{p}_k \in \mathbb{R}^{H_p \times W_p \times C}, \quad \text{typical } 1 \times 1 \times D$$

**Prototype assignment:**
$$d(\mathbf{z}_{patch}, \mathbf{p}_k) = \min_{j \in \text{patches}} \|\mathbf{z}_j - \mathbf{p}_k\|_2^2$$

### 2.3.2 Prototype Projection

For each prototype, compute the projection of embedding patches onto the prototype:

$$s_k = \max_{\text{patch } j} \mathbf{z}_j^\top \mathbf{p}_k$$

**Classification:**
$$P(y = k \mid \mathbf{x}) = \frac{\exp(s_k)}{\sum_{j} \exp(s_j)}$$

### 2.3.3 Interpretability Constraints

- **Cluster forcing:** Push prototype activations to be close to support set patches
- **Separation:** Enforce distance between prototypes of different classes
- **Prototype pruning:** Remove prototypes with low activation across dataset

## 2.4 Phytopathological Symptom Mapping

| Disease Class | Visual Symptom | Expected Heatmap Activation |
|---------------|----------------|----------------------------|
| Anthracnose | Circular dark lesions with acervuli | Concentrated necrotic spots |
| Bacterial Wilt | Wilted vines, brown vascular bundles | Vascular streak patterns |
| Belly Rot | Water-soaked lesions on fruit underside | Circular decay patterns |
| Downy Mildew | Angular yellow spots on leaf undersides | Angular chlorotic regions |
| Pythium Fruit Rot | Water-soaked, soft rotting | Diffuse decay areas |
| Gummy Stem Blight | Cankers with gummy exudate | Canker boundary regions |
| Fresh Leaves | Healthy green chlorophyll | Uniform leaf distribution |
| Fresh Cucumber | Uniform green skin | Full fruit surface |

---

# 3. EXPERIMENTAL SETUP & EVALUATION METRICS

## 3.1 Dataset Split Configuration

| Split | Classes | Images per Class | Total Images |
|-------|---------|------------------|--------------|
| Training | 6 classes | 128 (80%) | 768 |
| Validation | 2 classes | 32 (20%) | 64 |
| Test (Episode) | 8 classes (full) | 160 | 1,280 |

**Note:** For few-shot evaluation, test episodes sample from all 8 classes but support images are limited to $K \in \{1, 5\}$.

## 3.2 Episodic Evaluation Protocol

**Configuration 1: 5-way 1-shot**
- Classes per episode: 5
- Support samples per class: 1
- Query samples per class: 15
- Episodes: 1,000 randomized episodes
- Total query evaluations: 15,000

**Configuration 2: 5-way 5-shot**
- Classes per episode: 5
- Support samples per class: 5
- Query samples per class: 15
- Episodes: 1,000 randomized episodes
- Total query evaluations: 15,000

**Evaluation metric:**
$$\text{Accuracy} = \frac{\text{Number of Correct Query Predictions}}{\text{Total Number of Query Predictions}} \times 100\%$$

## 3.3 Baseline Comparisons

| Method | Architecture | Reference |
|--------|--------------|-----------|
| Matching Networks | 4-Conv embedding + LSTM attention | Vinyals et al., NIPS 2016 |
| Relation Networks | Conv-4 + relation module | Sung et al., CVPR 2018 |
| MAML (Model-Agnostic) | 4-Conv + FC | Finn et al., ICML 2017 |
| ProtoNet (baseline) | ResNet-18 | Snell et al., NIPS 2017 |
| ProtoNet + Grad-CAM | ResNet-18 + XAI | Proposed |

## 3.4 XAI Evaluation Metrics

### 3.4.1 Deletion Metric

Progressive masking of heatmap regions from highest to lowest importance:
1. Generate binary mask $M_t$ at threshold $t$ (top $t$\% important pixels)
2. Mask input: $\mathbf{x}_{masked} = \mathbf{x} \odot (1 - M_t)$
3. Evaluate probability drop: $\Delta P = P(y_{true}) - P(y_{true} \mid \mathbf{x}_{masked})$

**Score:** Area under the probability drop curve (AUC). Lower AUC indicates better localization.

### 3.4.2 Insertion Metric

Progressive addition of heatmap regions from highest to lowest importance:
1. Start with blurred/baseline image
2. Add pixels according to importance ranking
3. Track probability increase

**Score:** Area under the probability rise curve (AUC). Higher AUC indicates better localization.

### 3.4.3 Pointing Game

Quantitative localization accuracy:
1. Determine heatmap maximum point $(\hat{i}, \hat{j})$
2. Check if point falls within ground truth bounding box / region
3. Hit rate: $\frac{\text{Number of Hits}}{\text{Total Test Images}} \times 100\%$

### 3.4.4 Correlation with Expert Annotation

For curated subset with agronomist-annotated symptom bounding boxes:
- IoU (Intersection over Union) between heatmap and ground truth
- Precision@K: Fraction of top-K heatmap pixels within symptom region

---

# 4. PRODUCTION-READY PYTORCH PIPELINE

## 4.1 Core Module: PrototypicalNet

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np


class PrototypicalNet(nn.Module):
    """
    Prototypical Network for Few-Shot Learning with backbone encoder.
    
    Architecture:
    - Backbone: ResNet-18 or MobileNetV3
    - Prototype computation: Mean of support set embeddings
    - Distance metric: Squared Euclidean distance
    - Classification: Softmax over negative distances
    """
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        num_classes: int = 8,
        embedding_dim: int = 512,
        use_pretrained: bool = True
    ):
        """
        Initialize Prototypical Network.
        
        Args:
            backbone: Architecture choice ('resnet18', 'resnet50', 'mobilenet_v3_small', 'mobilenet_v3_large')
            num_classes: Total number of classes in dataset (for classifier layer)
            embedding_dim: Dimension of the embedding space
            use_pretrained: Whether to load ImageNet pretrained weights
        """
        super(PrototypicalNet, self).__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
        # Build backbone encoder
        self.encoder = self._build_encoder(backbone, use_pretrained)
        
        # Adaptive pooling to ensure consistent output dimension
        if 'mobilenet' in backbone:
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def _build_encoder(self, backbone: str, use_pretrained: bool) -> nn.Module:
        """Build and return the backbone encoder."""
        
        if backbone == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
            encoder = resnet18(weights=weights)
            # Remove final FC layer, keep avgpool output (512-dim)
            encoder = nn.Sequential(*list(encoder.children())[:-1])
            self.embedding_dim = 512
            
        elif backbone == 'resnet50':
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V1 if use_pretrained else None
            encoder = resnet50(weights=weights)
            encoder = nn.Sequential(*list(encoder.children())[:-1])
            self.embedding_dim = 2048
            
        elif backbone == 'mobilenet_v3_small':
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if use_pretrained else None
            encoder = mobilenet_v3_small(weights=weights)
            # Get features before final classifier
            encoder = nn.Sequential(
                encoder.features,
                encoder.avgpool
            )
            self.embedding_dim = 576
            
        elif backbone == 'mobilenet_v3_large':
            from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if use_pretrained else None
            encoder = mobilenet_v3_large(weights=weights)
            encoder = nn.Sequential(
                encoder.features,
                encoder.avgpool
            )
            self.embedding_dim = 960
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        return encoder
    
    def forward(
        self,
        x: torch.Tensor,
        prototypes: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Prototypical Network.
        
        Args:
            x: Input images [Batch, Channels, Height, Width] e.g., [N, 3, 224, 224]
            prototypes: Class prototypes [num_classes, embedding_dim] (optional, computed from support set)
        
        Returns:
            Tuple of (embeddings, logits):
                - embeddings: [Batch, embedding_dim]
                - logits: [Batch, num_classes]
        """
        # Encode input images to embeddings
        # Input: [Batch, 3, 224, 224]
        embeddings = self.encoder(x)  # [Batch, embedding_dim, 1, 1]
        
        # Flatten spatial dimensions
        embeddings = embeddings.view(embeddings.size(0), -1)  # [Batch, embedding_dim]
        
        # If prototypes not provided, return embeddings only
        if prototypes is None:
            return embeddings, None
        
        # Compute squared Euclidean distances to prototypes
        # prototypes: [num_classes, embedding_dim]
        # embeddings: [Batch, embedding_dim]
        
        # Compute distances: ||z - c||^2 = ||z||^2 + ||c||^2 - 2*z*c
        embeddings_squared = torch.sum(embeddings ** 2, dim=1, keepdim=True)  # [Batch, 1]
        prototypes_squared = torch.sum(prototypes ** 2, dim=1, keepdim=True)  # [num_classes, 1]
        
        cross_term = torch.matmul(embeddings, prototypes.t())  # [Batch, num_classes]
        
        distances = embeddings_squared + prototypes_squared.t() - 2 * cross_term
        # distances: [Batch, num_classes]
        
        # Compute logits as negative distances
        logits = -distances
        
        # Return probabilities via softmax
        probs = F.softmax(logits, dim=1)
        
        return embeddings, probs
    
    def compute_prototypes(
        self,
        support_embeddings: torch.Tensor,
        support_labels: torch.Tensor,
        num_classes: int
    ) -> torch.Tensor:
        """
        Compute class prototypes from support set embeddings.
        
        Args:
            support_embeddings: Embeddings of support set [N_support, embedding_dim]
            support_labels: Labels for support set [N_support]
            num_classes: Number of classes in the episode
        
        Returns:
            prototypes: [num_classes, embedding_dim]
        """
        prototypes = torch.zeros(
            num_classes,
            support_embeddings.size(1),
            device=support_embeddings.device,
            dtype=support_embeddings.dtype
        )
        
        for k in range(num_classes):
            # Get indices of support samples belonging to class k
            class_mask = (support_labels == k)
            class_embeddings = support_embeddings[class_mask]
            
            # Compute mean (prototype) for class k
            prototypes[k] = class_embeddings.mean(dim=0)
        
        return prototypes
    
    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate feature maps for Grad-CAM visualization.
        
        Args:
            x: Input images [Batch, 3, 224, 224]
        
        Returns:
            Feature maps from final convolutional layer
        """
        # Register hook to capture intermediate activations
        features = None
        
        def hook_fn(module, input, output):
            nonlocal features
            features = output
        
        # Find the final conv layer
        if 'resnet' in self.backbone_name:
            # Get the last conv layer in conv4_x (before final avgpool)
            final_conv = self.encoder[-2][-1].conv2
        elif 'mobilenet' in self.backbone_name:
            final_conv = self.encoder[0][-1].block[-1].conv2
        
        handle = final_conv.register_forward_hook(hook_fn)
        
        with torch.no_grad():
            _ = self.encoder(x)
        
        handle.remove()
        
        # For ResNet18: features shape [Batch, 512, 14, 14]
        # For MobileNetV3: depends on architecture
        return features
```

## 4.2 Episodic Batch Construction

```python
class EpisodicBatchSampler:
    """
    Sampler for creating N-way K-shot episodic batches.
    
    Generates support and query sets for few-shot learning episodes.
    """
    
    def __init__(
        self,
        labels: torch.Tensor,
        num_classes: int,
        num_ways: int,
        num_shots: int,
        num_queries: int = 15,
        episodes_per_epoch: int = 100
    ):
        """
        Initialize episodic batch sampler.
        
        Args:
            labels: All labels in dataset [N_samples]
            num_classes: Total number of classes in dataset
            num_ways: N-way (number of classes per episode)
            num_shots: K-shot (support samples per class)
            num_queries: Query samples per class
            episodes_per_epoch: Number of episodes to generate per epoch
        """
        self.labels = labels
        self.num_classes = num_classes
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries
        self.episodes_per_epoch = episodes_per_epoch
        
        # Build class to indices mapping
        self.class_to_indices = self._build_class_mapping()
        
    def _build_class_mapping(self) -> dict:
        """Map each class to its sample indices."""
        class_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label.item() not in class_to_indices:
                class_to_indices[label.item()] = []
            class_to_indices[label.item()].append(idx)
        return class_to_indices
    
    def __iter__(self):
        """Generate episodic batches."""
        for _ in range(self.episodes_per_epoch):
            # Randomly select N classes for this episode
            selected_classes = torch.randperm(self.num_classes)[:self.num_ways]
            
            support_indices = []
            query_indices = []
            
            for class_idx in selected_classes:
                # Get all samples for this class
                class_samples = self.class_to_indices[class_idx.item()]
                
                # Randomly sample K + Q samples (support + query)
                sampled_indices = torch.randperm(len(class_samples))[:self.num_shots + self.num_queries]
                
                # First K for support, remaining for query
                support_indices.extend([class_samples[i] for i in sampled_indices[:self.num_shots]])
                query_indices.extend([class_samples[i] for i in sampled_indices[self.num_shots:]])
            
            yield support_indices, query_indices
    
    def __len__(self):
        return self.episodes_per_epoch


class EpisodeCollator:
    """
    Collator for batching support and query sets with proper label reassignment.
    """
    
    def __init__(self, num_ways: int, num_shots: int, num_queries: int):
        self.num_ways = num_ways
        self.num_shots = num_shots
        self.num_queries = num_queries
    
    def __call__(self, batch: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate function for DataLoader.
        
        Args:
            batch: List of (image, original_label, is_query) tuples
        
        Returns:
            Tuple of (support_images, support_labels, query_images, query_labels)
            - support_images: [N_way * N_shot, C, H, W]
            - support_labels: [N_way * N_shot] (remapped to 0 to N-1)
            - query_images: [N_way * N_queries, C, H, W]
            - query_labels: [N_way * N_queries] (remapped to 0 to N-1)
        """
        # Separate support and query
        support_items = [item for item in batch if not item[2]]
        query_items = [item for item in batch if item[2]]
        
        # Stack support set
        support_images = torch.stack([item[0] for item in support_items])
        support_original_labels = torch.tensor([item[1] for item in support_items])
        
        # Stack query set
        query_images = torch.stack([item[0] for item in query_items])
        query_original_labels = torch.tensor([item[1] for item in query_items])
        
        # Remap labels to 0 to N-1 for episodic training
        unique_classes = torch.unique(support_original_labels)
        label_map = {old.item(): new for new, old in enumerate(unique_classes)}
        
        support_labels = torch.tensor([label_map[label.item()] for label in support_original_labels])
        query_labels = torch.tensor([label_map[label.item()] for label in query_original_labels])
        
        return support_images, support_labels, query_images, query_labels
```

## 4.3 Grad-CAM Integration

```python
class GradCAMExtractor:
    """
    Grad-CAM extractor for Prototypical Networks.
    
    Captures gradients from final convolutional layer to generate
    class-discriminative attention heatmaps.
    """
    
    def __init__(self, model: PrototypicalNet, target_layer_name: str = None):
        """
        Initialize Grad-CAM extractor.
        
        Args:
            model: Prototypical Network model
            target_layer_name: Name of target convolutional layer
        """
        self.model = model
        self.target_layer = None
        self.gradients = None
        self.activations = None
        
        self._find_target_layer(target_layer_name)
        self._register_hooks()
    
    def _find_target_layer(self, layer_name: str):
        """Locate the target convolutional layer for Grad-CAM."""
        if layer_name is not None:
            # User-specified layer
            for name, module in self.model.named_modules():
                if name == layer_name:
                    self.target_layer = module
                    return
            raise ValueError(f"Layer {layer_name} not found in model")
        
        # Auto-detect final conv layer
        if 'resnet' in self.model.backbone_name:
            # conv4_x final conv in ResNet
            self.target_layer = self.model.encoder[-2][-1].conv2
        elif 'mobilenet' in self.model.backbone_name:
            # Last conv in MobileNetV3
            self.target_layer = self.model.encoder[0][-1].block[-1].conv2
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_gradcam(
        self,
        query_image: torch.Tensor,
        prototypes: torch.Tensor,
        target_class: int
    ) -> torch.Tensor:
        """
        Generate Grad-CAM heatmap for a query image.
        
        Args:
            query_image: Single query image [1, C, H, W] or [C, H, W]
            prototypes: Class prototypes [num_ways, embedding_dim]
            target_class: Target class index for visualization
        
        Returns:
            heatmap: [H, W] normalized heatmap
        """
        # Ensure query_image has batch dimension
        if query_image.dim() == 3:
            query_image = query_image.unsqueeze(0)  # [1, C, H, W]
        
        # Set model to evaluation mode but enable gradients
        self.model.eval()
        
        # Clear stored gradients
        self.model.zero_grad()
        
        # Enable gradient computation for input (for visualization)
        query_image.requires_grad = True
        
        # Forward pass: compute embeddings and distances
        embeddings = self.model.encoder(query_image)  # [1, embedding_dim, 1, 1]
        embeddings = embeddings.view(embeddings.size(0), -1)  # [1, embedding_dim]
        
        # Compute squared Euclidean distances
        embeddings_sq = torch.sum(embeddings ** 2, dim=1, keepdim=True)  # [1, 1]
        prototypes_sq = torch.sum(prototypes ** 2, dim=1, keepdim=True)  # [num_ways, 1]
        cross_term = torch.matmul(embeddings, prototypes.t())  # [1, num_ways]
        
        distances = embeddings_sq + prototypes_sq.t() - 2 * cross_term  # [1, num_ways]
        
        # Get logit for target class
        target_logit = distances[0, target_class]
        
        # Backward pass
        target_logit.backward()
        
        # Get gradients and activations
        gradients = self.gradients  # [1, C, H', W']
        activations = self.activations  # [1, C, H', W']
        
        # Global average pooling of gradients to get weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [1, 1, H', W']
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Remove batch dimension and normalize
        cam = cam.squeeze().cpu().numpy()  # [H', W']
        
        # Normalize to [0, 1]
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def generate_multi_class_gradcam(
        self,
        query_image: torch.Tensor,
        prototypes: torch.Tensor
    ) -> dict:
        """
        Generate Grad-CAM heatmaps for all classes in the episode.
        
        Args:
            query_image: Query image [C, H, W] or [1, C, H, W]
            prototypes: Class prototypes [num_ways, embedding_dim]
        
        Returns:
            Dictionary mapping class indices to heatmaps
        """
        heatmaps = {}
        
        for class_idx in range(prototypes.size(0)):
            heatmaps[class_idx] = self.generate_gradcam(
                query_image,
                prototypes,
                target_class=class_idx
            )
        
        return heatmaps


def visualize_gradcam(
    image: torch.Tensor,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    save_path: str = None
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        image: Original image tensor [C, H, W]
        heatmap: Normalized heatmap [H, W]
        alpha: Transparency factor for overlay
        save_path: Optional path to save visualization
    
    Returns:
        overlayed_image: RGB numpy array
    """
    import cv2
    from PIL import Image
    
    # Convert tensor to numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # Denormalize if needed (ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    image = image * std + mean
    image = np.clip(image, 0, 1)
    image = (image.transpose(1, 2, 0) * 255).astype(np.uint8)  # [H, W, 3]
    
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    if save_path is not None:
        Image.fromarray(overlayed).save(save_path)
    
    return overlayed
```

## 4.4 Complete Training Pipeline

```python
class PrototypicalLoss(nn.Module):
    """
    Prototypical Network loss for few-shot learning.
    """
    
    def __init__(self, distance: str = 'euclidean'):
        super(PrototypicalLoss, self).__init__()
        self.distance = distance
    
    def forward(
        self,
        query_embeddings: torch.Tensor,
        prototypes: torch.Tensor,
        query_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute prototypical loss for query set.
        
        Args:
            query_embeddings: Query set embeddings [N_query, embedding_dim]
            prototypes: Class prototypes [N_way, embedding_dim]
            query_labels: Query set labels [N_query]
        
        Returns:
            loss: Scalar loss value
        """
        # Compute distances
        # query_embeddings: [N_query, D]
        # prototypes: [N_way, D]
        
        # Squared Euclidean: ||z - c||^2
        query_sq = torch.sum(query_embeddings ** 2, dim=1, keepdim=True)  # [N_query, 1]
        proto_sq = torch.sum(prototypes ** 2, dim=1, keepdim=True)  # [N_way, 1]
        cross = torch.matmul(query_embeddings, prototypes.t())  # [N_query, N_way]
        
        distances = query_sq + proto_sq.t() - 2 * cross  # [N_query, N_way]
        
        # Negative distance as logits (higher negative distance = closer = higher logit)
        logits = -distances
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, query_labels)
        
        return loss


def train_episode(
    model: PrototypicalNet,
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
    criterion: PrototypicalLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> dict:
    """
    Train one episode of prototypical network.
    
    Args:
        model: Prototypical Network
        support_images: Support set images [N_way * N_shot, C, H, W]
        support_labels: Support set labels [N_way * N_shot]
        query_images: Query set images [N_way * N_query, C, H, W]
        query_labels: Query set labels [N_way * N_query]
        criterion: Loss function
        optimizer: Optimizer
        device: Computation device
    
    Returns:
        Dictionary with loss and accuracy
    """
    # Move data to device
    support_images = support_images.to(device)
    support_labels = support_labels.to(device)
    query_images = query_images.to(device)
    query_labels = query_labels.to(device)
    
    # Get number of classes in this episode
    num_ways = len(torch.unique(support_labels))
    
    # Encode support set
    support_embeddings, _ = model(support_images)  # [N_way * N_shot, D]
    
    # Compute prototypes from support set
    prototypes = model.compute_prototypes(
        support_embeddings,
        support_labels,
        num_ways
    )  # [N_way, D]
    
    # Encode query set
    query_embeddings, _ = model(query_images)  # [N_way * N_query, D]
    
    # Compute loss
    loss = criterion(query_embeddings, prototypes, query_labels)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Compute accuracy
    with torch.no_grad():
        # Compute distances
        query_sq = torch.sum(query_embeddings ** 2, dim=1, keepdim=True)
        proto_sq = torch.sum(prototypes ** 2, dim=1, keepdim=True)
        cross = torch.matmul(query_embeddings, prototypes.t())
        distances = query_sq + proto_sq.t() - 2 * cross
        
        # Predictions
        predictions = torch.argmin(distances, dim=1)
        accuracy = (predictions == query_labels).float().mean().item()
    
    return {
        'loss': loss.item(),
        'accuracy': accuracy
    }


def evaluate_episode(
    model: PrototypicalNet,
    support_images: torch.Tensor,
    support_labels: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
    criterion: PrototypicalLoss,
    device: torch.device,
    gradcam_extractor: GradCAMExtractor = None
) -> dict:
    """
    Evaluate one episode with optional Grad-CAM generation.
    
    Args:
        model: Prototypical Network
        support_images: Support set images [N_way * N_shot, C, H, W]
        support_labels: Support set labels [N_way * N_shot]
        query_images: Query set images [N_way * N_query, C, H, W]
        query_labels: Query set labels [N_way * N_query]
        criterion: Loss function
        device: Computation device
        gradcam_extractor: Optional Grad-CAM extractor
    
    Returns:
        Dictionary with loss, accuracy, and optional heatmaps
    """
    model.eval()
    
    # Move to device
    support_images = support_images.to(device)
    support_labels = support_labels.to(device)
    query_images = query_images.to(device)
    query_labels = query_labels.to(device)
    
    num_ways = len(torch.unique(support_labels))
    
    with torch.no_grad():
        # Encode support and compute prototypes
        support_embeddings, _ = model(support_images)
        prototypes = model.compute_prototypes(support_embeddings, support_labels, num_ways)
        
        # Encode queries
        query_embeddings, probs = model(query_images, prototypes)
        
        # Loss
        loss = criterion(query_embeddings, prototypes, query_labels)
        
        # Predictions
        predictions = torch.argmax(probs, dim=1)
        accuracy = (predictions == query_labels).float().mean().item()
    
    result = {
        'loss': loss.item(),
        'accuracy': accuracy,
        'predictions': predictions.cpu(),
        'true_labels': query_labels.cpu()
    }
    
    # Generate Grad-CAM for first query if extractor provided
    if gradcam_extractor is not None:
        heatmaps = []
        for i in range(query_images.size(0)):
            hm = gradcam_extractor.generate_gradcam(
                query_images[i],
                prototypes,
                target_class=predictions[i].item()
            )
            heatmaps.append(hm)
        result['heatmaps'] = heatmaps
    
    return result
```

## 4.5 Complete Training Loop

```python
def main():
    """
    Main training and evaluation loop for Prototypical Network.
    """
    import argparse
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from PIL import Image
    import os
    
    # Configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_ways', type=int, default=5, help='N-way')
    parser.add_argument('--num_shots', type=int, default=1, help='K-shot')
    parser.add_argument('--num_queries', type=int, default=15, help='Query samples per class')
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--eval_episodes', type=int, default=1000)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Initialize model
    model = PrototypicalNet(backbone=args.backbone).to(device)
    criterion = PrototypicalLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Grad-CAM extractor for evaluation
    gradcam_extractor = GradCAMExtractor(model)
    
    # Training loop
    for epoch in range(args.num_epochs):
        epoch_loss = 0
        epoch_acc = 0
        
        # Example: iterate over episodic batches
        for episode_idx in range(args.episodes_per_epoch):
            # Sample support and query sets (pseudo-code - implement with dataset)
            # In practice, use EpisodicBatchSampler with dataset
            support_images, support_labels, query_images, query_labels = sample_episode()
            
            result = train_episode(
                model, support_images, support_labels,
                query_images, query_labels,
                criterion, optimizer, device
            )
            
            epoch_loss += result['loss']
            epoch_acc += result['accuracy']
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {epoch_loss/args.episodes_per_epoch:.4f}, "
              f"Acc: {epoch_acc/args.episodes_per_epoch:.4f}")
    
    # Evaluation
    eval_accuracies = []
    for _ in range(args.eval_episodes):
        support_images, support_labels, query_images, query_labels = sample_episode()
        
        result = evaluate_episode(
            model, support_images, support_labels,
            query_images, query_labels,
            criterion, device
        )
        
        eval_accuracies.append(result['accuracy'])
    
    print(f"5-way {args.num_shots}-shot Accuracy: {np.mean(eval_accuracies)*100:.2f}% ± {np.std(eval_accuracies)*100:.2f}%")
```

---

# 5. EXPECTED ABLATION STUDIES & DISCUSSION OUTLINE

## 5.1 Ablation Study: Backbone Comparison

| Backbone | Parameters (M) | 5-way 1-shot Acc (%) | 5-way 5-shot Acc (%) | Inference Time (ms) | Grad-CAM Quality |
|----------|----------------|---------------------|---------------------|---------------------|------------------|
| ResNet-18 | 11.2 | TBD | TBD | ~15 | Good |
| ResNet-50 | 25.6 | TBD | TBD | ~25 | Better |
| MobileNetV3-Small | 2.5 | TBD | TBD | ~8 | Moderate |
| MobileNetV3-Large | 5.5 | TBD | TBD | ~12 | Good |

**Analysis:** Trade-off between computational efficiency and representation capacity. MobileNetV3-Small suitable for edge deployment; ResNet-50 for maximum accuracy.

## 5.2 Ablation Study: Embedding Dimension

| Embedding Dim | 5-way 1-shot Acc (%) | 5-way 5-shot Acc (%) | Overfitting Risk |
|---------------|---------------------|---------------------|------------------|
| 128 | TBD | TBD | Low |
| 256 | TBD | TBD | Low-Moderate |
| 512 | TBD | TBD | Moderate |
| 1024 | TBD | TBD | High |

**Analysis:** Lower dimensions may underfit complex disease phenotypes; higher dimensions risk overfitting given limited K-shot support.

## 5.3 Ablation Study: Distance Metric

| Metric | Formula | 5-way 1-shot Acc (%) | 5-way 5-shot Acc (%) |
|--------|---------|---------------------|---------------------|
| Squared Euclidean | $\|\mathbf{z} - \mathbf{c}\|_2^2$ | TBD | TBD |
| Euclidean | $\|\mathbf{z} - \mathbf{c}\|_2$ | TBD | TBD |
| Cosine Similarity | $\frac{\mathbf{z} \cdot \mathbf{c}}{\|\mathbf{z}\|\|\mathbf{c}\|}$ | TBD | TBD |
| Mahalanobis | $(\mathbf{z} - \mathbf{c})^\top \Sigma^{-1} (\mathbf{z} - \mathbf{c})$ | TBD | TBD |

**Analysis:** Euclidean captures angular relationships in embedding space; Cosine focuses on direction. Mahalanobis adapts to class covariance structure but increases computational overhead.

## 5.4 Ablation Study: XAI Method Comparison

| Method | Deletion AUC (↓) | Insertion AUC (↑) | Pointing Game (%) | Computational Cost |
|--------|-----------------|-------------------|-------------------|-------------------|
| Grad-CAM | TBD | TBD | TBD | Low |
| Integrated Gradients | TBD | TBD | TBD | High |
| Grad-CAM++ | TBD | TBD | TBD | Moderate |
| Score-CAM | TBD | TBD | TBD | Moderate |

## 5.5 Discussion: Phytopathological Alignment

### 5.5.1 Bridging Computer Vision and Agronomic Expertise

**Key Discussion Points:**

1. **Symptom Localization Accuracy**
   - How precisely do Grad-CAM heatmaps align with agronomist-identified disease regions?
   - Correlation between model attention and expert-annotated bounding boxes
   - Implications for extension services: Can farmers trust model explanations?

2. **Interpretability vs. Accuracy Trade-off**
   - Does intrinsic interpretability (ProtoPNet) sacrifice classification accuracy?
   - User study: Would agronomists prefer slightly lower accuracy with transparent reasoning?

3. **Transfer Learning Considerations**
   - ImageNet pretraining captures generic visual features (edges, textures, shapes)
   - Domain gap between natural images and leaf pathology images
   - Potential for agricultural-pretrained encoders (e.g., PlantNet, AgriNet)

4. **Practical Deployment Scenarios**
   - Mobile field diagnostics: Model inference on smartphone
   - Integration with agricultural extension workflow
   - Trust calibration: How to present confidence alongside explanations

5. **Failure Mode Analysis**
   - What happens when model confuses similar diseases (e.g., Downy Mildew vs. Angular Leaf Spot)?
   - Can XAI heatmaps reveal why these errors occur?
   - Implications for dataset curation: Which confusing pairs need more samples?

### 5.5.2 Societal Impact

- **Precision Agriculture:** Enabling low-resource farmers to access expert-level diagnosis
- **Sustainability:** Early disease detection reduces pesticide usage
- **Education:** Training next-generation agronomists with AI-assisted tools

---

# 6. MANUSCRIPT STRUCTURE RECOMMENDATION

## IEEE Access / Computers and Electronics in Agriculture Format

| Section | Content | Target Length |
|---------|---------|---------------|
| Abstract | Problem, method, results, impact | 150-250 words |
| Introduction | FSL in agriculture, XAI motivation, contributions | 1.5-2 pages |
| Related Work | FSL methods, XAI in agriculture, plant disease datasets | 1-1.5 pages |
| Proposed Method | Mathematical formulation, architecture, XAI integration | 2-3 pages |
| Experimental Setup | Dataset, evaluation protocol, baselines | 1-1.5 pages |
| Results | Main results, ablation studies, XAI evaluation | 2-2.5 pages |
| Discussion | Phytopathological alignment, practical implications | 1-1.5 pages |
| Conclusion | Summary, limitations, future work | 0.5-1 page |
| References | 30-50 citations | - |

---

*This document serves as a comprehensive technical blueprint for implementing Few-Shot Explainable Plant Pathology using Prototypical Networks with pixel-level attribution. The framework is designed for reproducibility and modularity in PyTorch, targeting peer-reviewed publication in Q1 agricultural/computer vision journals.*