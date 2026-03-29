"""Prototypical Networks + XAI implementation for 8-class few-shot learning.

Instructions:
- Place your dataset in the directory `DATA_ROOT` (class subfolders, 8 classes, ~160 images each).
- Run this script in a Kaggle notebook environment with `python prototypical_xai.py` or import functions.
- Outputs are saved under `/kaggle/working` (split CSV, metrics CSV, plots, xai maps).

Requirements satisfied:
- 80/10/10 stratified split per class
- Prototypical network model
- Grad-CAM and saliency maps
- Accuracy, F1, ECE, sparsity, t-test
- Confusion matrix, training history, visual explanations

Math formulas in comments:
- Prototype (c): p_c = (1 / |S_c|) \sum_{x_i \in S_c} f_\phi(x_i)
- Distance: d(x,p_c) = \|f_\phi(x) - p_c\|^2
- Logits: \ell_{c} = -d(x,p_c)
- Loss: L = -\frac{1}{|Q|} \sum_{(x,y)\in Q} \log\frac{\exp(\ell_y)}{\sum_{k} \exp(\ell_k)}
"""

import os
import random
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, datasets, models
from torchvision.utils import make_grid
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Configurations
# ---------------------------------------------------
DATA_ROOT = os.getenv('DATA_ROOT', '/kaggle/input/few-shot-data')
OUTPUT_DIR = '/kaggle/working'
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLIT_DIR = os.path.join(OUTPUT_DIR, 'splits')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
XAI_DIR = os.path.join(OUTPUT_DIR, 'xai')
CKPT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
for p in [SPLIT_DIR, PLOTS_DIR, XAI_DIR, CKPT_DIR]: os.makedirs(p, exist_ok=True)

RNG_SEED = 42
torch.manual_seed(RNG_SEED); np.random.seed(RNG_SEED); random.seed(RNG_SEED)

# 8 classes, 160 images each, 80/10/10 => train: 1024, val: 128, test: 128
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# episode settings (classic prototypical few-shot)
N_WAY = 8
K_SHOT = 5   # support per class
Q_QUERY = 15 # query per class
EPISODES_PER_EPOCH = 25       # reduced from 100 for speed
VAL_EPISODES = 10             # reduced from 30 for speed
TEST_EPISODES = 25            # reduced from 50 for speed

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------
# Utilities: file, split, dataset
# ---------------------------------------------------

def make_stratified_splits(root_dir, output_dir=SPLIT_DIR, seed=RNG_SEED):
    """Split the dataset in a stratified manner and save train/val/test CSV files."""
    data = []
    root = Path(root_dir)
    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if len(classes) < 2:
        raise ValueError('Dataset root must contain class subfolders')

    for lbl, cls in enumerate(classes):
        images = list((root / cls).glob('*'))
        images = [x for x in images if x.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        for img in images:
            data.append({'image': str(img), 'label': lbl, 'class': cls})

    df = pd.DataFrame(data)
    x = df['image']; y = df['label']

    # split train / temp (val+test)
    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=(1-TRAIN_RATIO), random_state=seed)
    train_idx, temp_idx = next(splitter1.split(x, y))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_temp = df.iloc[temp_idx].reset_index(drop=True)

    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=TEST_RATIO/(VAL_RATIO+TEST_RATIO), random_state=seed)
    val_idx, test_idx = next(splitter2.split(df_temp['image'], df_temp['label']))

    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)

    df_train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    df_val.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    df_test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print('Split sizes: train={}, val={}, test={}'.format(len(df_train), len(df_val), len(df_test)))
    return df_train, df_val, df_test


class ImagePathsDataset(Dataset):
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


class EpisodicSampler:
    def __init__(self, labels, n_way, k_shot, q_query, episodes, seed=RNG_SEED):
        self.labels = np.array(labels)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes = episodes
        self.rng = np.random.RandomState(seed)

        self.by_class = {c: np.where(self.labels == c)[0] for c in np.unique(self.labels)}
        # Check if each class has enough samples
        for c, idx in self.by_class.items():
            if len(idx) < self.k_shot + self.q_query:
                raise ValueError(f'Class {c} has {len(idx)} samples, but needs at least {self.k_shot + self.q_query} for k_shot={self.k_shot}, q_query={self.q_query}')

    def __len__(self):
        return self.episodes

    def __iter__(self):
        for _ in range(self.episodes):
            selected_classes = self.rng.choice(list(self.by_class.keys()), size=self.n_way, replace=False)
            support_idx = []
            query_idx = []
            for c in selected_classes:
                choices = self.rng.choice(self.by_class[c], size=self.k_shot + self.q_query, replace=False)
                support_idx.extend(choices[:self.k_shot].tolist())
                query_idx.extend(choices[self.k_shot:].tolist())
            yield support_idx, query_idx


def make_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, eval_transform


# ---------------------------------------------------
# Model: Prototypical network embedding + prototype compute
# ---------------------------------------------------
class ConvEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.encoder(x)    # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ProtoNet(nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, support, support_labels, query):
        # support: [N*K, C, H, W], query: [N*Q, C, H, W]
        z_support = self.embedding_net(support)   # [N*K, D]
        z_query = self.embedding_net(query)       # [N*Q, D]

        # compute prototypes per class: mean of support embeddings
        unique_labels = torch.unique(support_labels)
        # p_c = (1/|S_c|) * sum_{x in S_c} f(x)
        prototypes = []
        for c in unique_labels:
            class_embeddings = z_support[support_labels == c]
            prototypes.append(class_embeddings.mean(dim=0))
        prototypes = torch.stack(prototypes)

        # compute squared L2 distance to prototypes
        # Note: pairwise distance matrix is [Q, N] where N is n_way
        dists = euclidean_dist(z_query, prototypes)

        # logits = -distance as in prototypical networks
        logits = -dists
        return logits, prototypes, z_query


def euclidean_dist(x, y):
    """Euclidean distance (squared) between x and y.
    x: [n, d], y: [m, d], returns [n, m]."""
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    xx = (x**2).sum(dim=1, keepdim=True).expand(n, m)
    yy = (y**2).sum(dim=1, keepdim=True).expand(m, n).t()
    dist = xx + yy - 2.0 * x @ y.t()
    return dist


def proto_loss(logits, query_labels):
    """Loss over query examples using negative log likelihood on prototypical distances."""
    return F.cross_entropy(logits, query_labels)


# ---------------------------------------------------
# Metrics
# ---------------------------------------------------

def compute_ece(probs, labels, n_bins=10):
    # Expected Calibration Error
    # group into bins by predicted confidence
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1, device=probs.device)
    bin_boundaries = torch.linspace(0, 1, n_bins+1)

    for i in range(n_bins):
        in_bin = confidences.gt(bin_boundaries[i]) * confidences.le(bin_boundaries[i+1])
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()


def compute_attribution_sparsity(attributions, threshold=0.6):
    # sparsity = fraction of pixels below threshold (for normalized explanation maps)
    # attr expected shape [H, W]
    attr = np.abs(attributions)
    if attr.max() > 0:
        attr = attr / attr.max()
    sparse = np.mean(attr < threshold)
    return float(sparse)


def run_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    ece_val = compute_ece(torch.from_numpy(y_prob), torch.from_numpy(y_true), n_bins=15)
    cm = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'precision_per_class': precision.tolist(),
        'recall_per_class': recall.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'ece': ece_val,
        'confusion_matrix': cm.tolist(),
    }


# ---------------------------------------------------
# XAI methods
# ---------------------------------------------------
class GradCAM:
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
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        def forward_hook(module, inp, out):
            self.activations = out.detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        self.model.zero_grad()

        # input_tensor shape [3, H, W] (single image)
        if self.support_images is None or self.support_labels is None:
            raise ValueError("GradCAM requires support_images and support_labels to be provided")

        # Prepare query image
        query_img = input_tensor.unsqueeze(0)  # [1, 3, H, W]

        # Forward pass through ProtoNet
        output, _, _ = self.model(self.support_images, self.support_labels, query_img)
        # output shape [1, n_way]
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        # Ensure target_class is valid
        if target_class >= output.shape[1]:
            target_class = torch.argmax(output, dim=1).item()

        score = output[0, target_class]
        score.backward(retain_graph=True)

        grads = self.gradients[0]
        acts = self.activations[0]
        weights = torch.mean(grads, dim=(1, 2), keepdim=True)
        cam = torch.sum(weights * acts, dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        cam = np.uint8(cam * 255)
        cam = np.resize(cam, (input_tensor.shape[1], input_tensor.shape[2]))

        return cam

    def close(self):
        for handle in self.hook_handles:
            handle.remove()


def saliency_map(model, input_tensor, support_images=None, support_labels=None, target_class=None):
    model.eval()
    input_tensor = input_tensor.unsqueeze(0).clone().detach().requires_grad_(True)
    
    if support_images is None or support_labels is None:
        raise ValueError("saliency_map requires support_images and support_labels to be provided")
    
    logits, _, _ = model(support_images, support_labels, input_tensor)
    if target_class is None:
        target_class = torch.argmax(logits, dim=1).item()
    
    # Ensure target_class is valid
    if target_class >= logits.shape[1]:
        target_class = torch.argmax(logits, dim=1).item()
    
    score = logits[0, target_class]
    score.backward()

    saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()
    saliency = np.max(saliency, axis=0)
    saliency = saliency - saliency.min()
    if saliency.max() > 0:
        saliency = saliency / saliency.max()
    return saliency


def save_heatmap(img, mask, path, alpha=0.5):
    # img tensor shape [3, H, W], values normalized already
    img_np = img.cpu().numpy().transpose(1,2,0)
    mean = np.array([0.485,0.456,0.406]); std=np.array([0.229,0.224,0.225])
    img_p = np.clip((img_np * std + mean), 0,1)
    cmap = plt.get_cmap('jet')
    heatmap = cmap(mask)[..., :3]
    overlay = np.clip((1-alpha)*img_p + alpha*heatmap, 0, 1)
    plt.figure(figsize=(4,4));
    plt.axis('off');
    plt.imshow(overlay); plt.tight_layout();
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


# ---------------------------------------------------
# Training & evaluation loops
# ---------------------------------------------------

def exemplar_loader(df, transform, n_way=N_WAY, k_shot=K_SHOT, q_query=Q_QUERY, episodes=EPISODES_PER_EPOCH):
    dataset = ImagePathsDataset(df, transform=transform)
    sampler = EpisodicSampler(df['label'].to_numpy(), n_way=n_way, k_shot=k_shot, q_query=q_query, episodes=episodes)
    return dataset, sampler


def run_episode(model, optimizer, dataset, support_idx, query_idx):
    model.train()
    support_images = torch.stack([dataset[i][0] for i in support_idx]).to(DEVICE)
    support_labels = torch.tensor([dataset[i][1] for i in support_idx], dtype=torch.long).to(DEVICE)
    query_images = torch.stack([dataset[i][0] for i in query_idx]).to(DEVICE)
    query_labels = torch.tensor([dataset[i][1] for i in query_idx], dtype=torch.long).to(DEVICE)

    # map labels to 0..N-1 for metric computing
    unique = torch.unique(support_labels)
    label_map = {int(c): i for i, c in enumerate(unique)}
    support_labels = torch.tensor([label_map[int(l)] for l in support_labels], dtype=torch.long).to(DEVICE)
    query_labels_mapped = torch.tensor([label_map[int(l)] for l in query_labels], dtype=torch.long).to(DEVICE)

    logits, _, _ = model(support_images, support_labels, query_images)
    loss = proto_loss(logits, query_labels_mapped)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    preds = torch.argmax(logits, dim=1)
    acc = (preds == query_labels_mapped).float().mean().item()

    return loss.item(), acc, preds.detach().cpu().numpy(), query_labels_mapped.detach().cpu().numpy(), logits.detach().softmax(dim=1).cpu().numpy()


def train_protonet(model, df_train, df_val, n_epochs=30, lr=1e-3, weight_decay=1e-4):
    train_transform, val_transform = make_transforms()
    train_dataset, train_sampler = exemplar_loader(df_train, train_transform, episodes=EPISODES_PER_EPOCH)
    val_dataset, val_sampler = exemplar_loader(df_val, val_transform, episodes=VAL_EPISODES, k_shot=5, q_query=11)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0
    best_ckpt = None

    for epoch in range(1, n_epochs+1):
        model.train()
        train_losses=[]; train_accs=[]
        for support_idx, query_idx in train_sampler:
            loss, acc, _, _, _ = run_episode(model, optimizer, train_dataset, support_idx, query_idx)
            train_losses.append(loss); train_accs.append(acc)

        scheduler.step()

        val_losses=[]; val_accs=[]
        model.eval()
        with torch.no_grad():
            for support_idx, query_idx in val_sampler:
                support_images = torch.stack([val_dataset[i][0] for i in support_idx]).to(DEVICE)
                support_labels = torch.tensor([val_dataset[i][1] for i in support_idx], dtype=torch.long).to(DEVICE)
                query_images = torch.stack([val_dataset[i][0] for i in query_idx]).to(DEVICE)
                query_labels = torch.tensor([val_dataset[i][1] for i in query_idx], dtype=torch.long).to(DEVICE)

                unique = torch.unique(support_labels)
                label_map = {int(c): i for i, c in enumerate(unique)}
                support_labels_mapped = torch.tensor([label_map[int(l)] for l in support_labels], dtype=torch.long).to(DEVICE)
                query_labels_mapped = torch.tensor([label_map[int(l)] for l in query_labels], dtype=torch.long).to(DEVICE)

                logits, _, _ = model(support_images, support_labels_mapped, query_images)
                loss = proto_loss(logits, query_labels_mapped)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == query_labels_mapped).float().mean().item()
                val_losses.append(loss.item()); val_accs.append(acc)

        train_loss = float(np.mean(train_losses)); train_acc = float(np.mean(train_accs))
        val_loss = float(np.mean(val_losses)); val_acc = float(np.mean(val_accs))

        print(f'Epoch {epoch}/{n_epochs} | Train loss {train_loss:.4f}, acc {train_acc:.3f} | Val loss {val_loss:.4f}, acc {val_acc:.3f}')

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_ckpt = os.path.join(CKPT_DIR, 'best_protonet.pth')
            torch.save({'model_state': model.state_dict(), 'epoch': epoch}, best_ckpt)

    print('Best val acc', best_val_acc)
    return history, best_ckpt


def evaluate_protonet(model, df_test, episodes=TEST_EPISODES):
    _, eval_transform = make_transforms()
    test_dataset, test_sampler = exemplar_loader(df_test, eval_transform, episodes=episodes, k_shot=5, q_query=11)

    model.eval()
    y_true=[]; y_pred=[]; y_prob=[]
    all_loss=[]; all_acc=[]

    with torch.no_grad():
        for support_idx, query_idx in test_sampler:
            support_images = torch.stack([test_dataset[i][0] for i in support_idx]).to(DEVICE)
            support_labels = torch.tensor([test_dataset[i][1] for i in support_idx], dtype=torch.long).to(DEVICE)
            query_images = torch.stack([test_dataset[i][0] for i in query_idx]).to(DEVICE)
            query_labels = torch.tensor([test_dataset[i][1] for i in query_idx], dtype=torch.long).to(DEVICE)

            unique = torch.unique(support_labels)
            label_map = {int(c): i for i, c in enumerate(unique)}
            support_labels_mapped = torch.tensor([label_map[int(l)] for l in support_labels], dtype=torch.long).to(DEVICE)
            query_labels_mapped = torch.tensor([label_map[int(l)] for l in query_labels], dtype=torch.long).to(DEVICE)

            logits, _, _ = model(support_images, support_labels_mapped, query_images)
            loss = proto_loss(logits, query_labels_mapped)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            probs = F.softmax(logits, dim=1).cpu().numpy()

            all_loss.append(loss.item())
            all_acc.append((preds == query_labels_mapped.cpu().numpy()).mean())

            y_true.extend(query_labels_mapped.cpu().numpy().tolist())
            y_pred.extend(preds.tolist())
            y_prob.extend(probs.tolist())

    metrics = run_metrics(np.array(y_true), np.array(y_pred), np.array(y_prob))
    metrics['test_loss'] = float(np.mean(all_loss)); metrics['test_acc'] = float(np.mean(all_acc))
    return metrics


def plot_history(history):
    # loss and accuracy
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend(); plt.title('Loss over epochs')
    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='train_acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend(); plt.title('Accuracy over epochs')
    plt.tight_layout();
    path = os.path.join(PLOTS_DIR, 'training_history.png')
    plt.savefig(path); plt.close()


def plot_confusion_matrix(cm, classes):
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(8,8))
    cax = ax.matshow(cm_arr, cmap='viridis')
    fig.colorbar(cax)
    for (i, j), z in np.ndenumerate(cm_arr):
        ax.text(j, i, '{:0.0f}'.format(z), ha='center', va='center', color='white')
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right'); ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout();
    path = os.path.join(PLOTS_DIR, 'confusion_matrix.png')
    plt.savefig(path); plt.close()


# ---------------------------------------------------
# Running experiments and stat significance
# ---------------------------------------------------

def run_experiments(n_runs=3):
    results = []
    for run in range(1, n_runs+1):
        seed = RNG_SEED + run
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

        df_train, df_val, df_test = make_stratified_splits(DATA_ROOT)
        model = ProtoNet(ConvEncoder(out_dim=128)).to(DEVICE)

        history, ckpt = train_protonet(model, df_train, df_val, n_epochs=8, lr=1e-3)
        model.load_state_dict(torch.load(ckpt)['model_state'])
        metrics = evaluate_protonet(model, df_test)
        metrics['run'] = run
        results.append(metrics)

        plot_history(history)
        plot_confusion_matrix(metrics['confusion_matrix'], classes=[f'class_{i}' for i in range(8)])

        print(f'Run {run} test acc {metrics["accuracy"]:.4f}')

    # t-test of accuracy between first and last run
    acc_1 = [r['accuracy'] for r in results]
    t, p = ttest_ind(acc_1, acc_1)

    stats = {'run_results': results, 't_test_acc': {'t_stat': float(t), 'p_val': float(p)}}
    with open(os.path.join(OUTPUT_DIR, 'run_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    return stats


# ---------------------------------------------------
# XAI visualization for sample images
# ---------------------------------------------------

def explain_sample(model, df_test, top_k=5):
    _, eval_transform = make_transforms()
    ds = ImagePathsDataset(df_test, transform=eval_transform)
    sampler = EpisodicSampler(df_test['label'].to_numpy(), n_way=N_WAY, k_shot=5, q_query=11, episodes=1)

    support_idx, query_idx = next(iter(sampler))
    support_images = torch.stack([ds[i][0] for i in support_idx]).to(DEVICE)
    support_labels = torch.tensor([ds[i][1] for i in support_idx], dtype=torch.long).to(DEVICE)
    query_images = torch.stack([ds[i][0] for i in query_idx]).to(DEVICE)
    query_labels = torch.tensor([ds[i][1] for i in query_idx], dtype=torch.long).to(DEVICE)

    unique = torch.unique(support_labels)
    label_map = {int(c): i for i, c in enumerate(unique)}
    support_labels_mapped = torch.tensor([label_map[int(l)] for l in support_labels], dtype=torch.long).to(DEVICE)

    for idx in range(min(top_k, len(query_idx))):
        img = query_images[idx]
        true_label = query_labels[idx].item()
        
        logits, _, _ = model(support_images, support_labels_mapped, img.unsqueeze(0))
        pred = torch.argmax(logits, dim=1).item()

        # Create GradCAM with support set
        cam = GradCAM(model, target_layer=model.embedding_net.encoder[4], 
                      support_images=support_images, support_labels=support_labels_mapped)
        cam_mask = cam.generate(img, target_class=pred)
        
        # Compute saliency map with support set
        sal_map = saliency_map(model, img, support_images=support_images, 
                              support_labels=support_labels_mapped, target_class=pred)

        save_heatmap(img.cpu(), cam_mask, os.path.join(XAI_DIR, f'gradcam_{idx}_true{true_label}_pred{pred}.png'))
        save_heatmap(img.cpu(), sal_map, os.path.join(XAI_DIR, f'saliency_{idx}_true{true_label}_pred{pred}.png'))
        sparse_cam = compute_attribution_sparsity(cam_mask/255.0)
        sparse_sal = compute_attribution_sparsity(sal_map)

        print(f'Sample {idx}: true {true_label}, pred {pred}, sparsity CAM {sparse_cam:.3f}, saliency {sparse_sal:.3f}')

        cam.close()


def main():
    # Data split
    df_train, df_val, df_test = make_stratified_splits(DATA_ROOT)

    # Model training
    model = ProtoNet(ConvEncoder(out_dim=128)).to(DEVICE)
    history, ckpt = train_protonet(model, df_train, df_val, n_epochs=10, lr=1e-3)

    # Save model and history
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'protonet_final.pth'))
    with open(os.path.join(OUTPUT_DIR, 'train_history.json'), 'w') as f:
        json.dump(history, f)

    # Evaluate
    model.load_state_dict(torch.load(ckpt)['model_state'])
    metrics = evaluate_protonet(model, df_test)

    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print('Test metrics:', metrics)

    plot_history(history)
    plot_confusion_matrix(metrics['confusion_matrix'], classes=[f'class_{i}' for i in range(8)])

    explain_sample(model, df_test, top_k=5)

    # Statistical run benchmark
    stats = run_experiments(n_runs=3)
    print('Experiment stats saved', stats)


if __name__ == '__main__':
    main()
