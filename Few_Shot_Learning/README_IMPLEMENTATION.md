# Prototypical Networks with XAI for Few-Shot Learning

## Research-Grade Implementation Documentation

---

## Overview

This implementation provides a complete, research-ready solution for few-shot image classification using **Prototypical Networks** integrated with **Explainable AI (XAI)** techniques.

### Key Features

| Feature | Description |
|---------|-------------|
| **Prototypical Networks** | Metric-based few-shot learning (Snell et al., 2017) |
| **XAI Methods** | Grad-CAM, Saliency Maps, Integrated Gradients |
| **Evaluation Metrics** | Accuracy, F1-score, ECE, Brier Score, Attribution Sparsity |
| **Statistical Testing** | Multiple independent runs with t-tests |
| **Visualizations** | Training curves, confusion matrix, per-class metrics, calibration curves |
| **Kaggle-Ready** | Optimized for Kaggle's 15GB GPU environment |

---

## Mathematical Foundations

### 1. Prototype Computation

The prototype for each class is the mean of its support set embeddings:

```
p_c = (1 / |S_c|) Σ_{(x_i, y_i) ∈ S_c} f_φ(x_i)
```

Where:
- `p_c`: prototype for class c
- `S_c`: support set for class c
- `f_φ`: embedding function (CNN encoder)
- `|S_c|`: number of support samples (K-shot)

### 2. Distance Metric

Squared Euclidean distance between query and prototypes:

```
d(z, p_c) = ‖f_φ(z) - p_c‖²_2 = Σ_j (f_φ(z)_j - p_{c,j})²
```

### 3. Classification

Softmax over negative distances:

```
P(y = c | z) = exp(-d(z, p_c)) / Σ_{c'} exp(-d(z, p_{c'}))
```

### 4. Loss Function

Negative log-likelihood over query set:

```
L(φ) = - (1 / |Q|) Σ_{(z_j, y_j) ∈ Q} log P(y = y_j | z_j)
```

### 5. Evaluation Metrics

**Accuracy:**
```
ACC = (1/N) Σ_i 1(ŷ_i = y_i)
```

**F1-Score:**
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Expected Calibration Error:**
```
ECE = Σ_{m=1}^M (|B_m| / n) |acc(B_m) - conf(B_m)|
```

**Attribution Sparsity:**
```
S(A) = (1 / HW) Σ_{i,j} 1(|A_{i,j}| < τ)
```

---

## File Structure

```
Few_Shot_Learning/
├── prototypical_xai_complete.py      # Complete implementation (script)
├── prototypical_xai_kaggle.ipynb     # Jupyter notebook for Kaggle
├── requirements.txt                  # Python dependencies
├── README_IMPLEMENTATION.md           # This documentation
└── prototypical_xai/                  # Previous implementation
    ├── prototypical_xai.py
    └── ...
```

---

## Usage

### Option 1: Kaggle Notebook (Recommended)

1. Upload your dataset to `/kaggle/input/few-shot-data/`
   ```
   few-shot-data/
   ├── class_0/
   │   ├── img_001.jpg
   │   └── ...
   ├── class_1/
   └── ... (8 classes total)
   ```

2. Upload `prototypical_xai_kaggle.ipynb` to Kaggle

3. Run all cells

4. Download outputs from `/kaggle/working/`

### Option 2: Local/Server Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Run the script
python prototypical_xai_complete.py
```

### Option 3: Interactive Python

```python
from prototypical_xai_complete import main, Config

# Customize configuration
cfg = Config(
    data_root='path/to/data',
    n_epochs=50,
    lr=1e-4
)

# Run training and evaluation
results = main(cfg, run_experiments=False)

# For statistical significance testing
results = main(cfg, run_experiments=True, n_experiment_runs=3)
```

---

## Configuration

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_way` | 8 | Classes per episode |
| `k_shot` | 5 | Support samples per class |
| `q_query` | 15 | Query samples per class |
| `n_epochs` | 30 | Training epochs |
| `lr` | 1e-3 | Learning rate |
| `embedding_dim` | 128 | Embedding dimension |
| `episodes_per_epoch` | 20 | Episodes per training epoch |

### Memory Optimization

For Kaggle's 15GB GPU limit:
- Image size: 128×128 (configurable)
- Batch processing via episodes
- Gradient checkpointing (optional)
- Automatic mixed precision support

---

## XAI Methods

### 1. Grad-CAM

**Reference:** Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017

**Formula:**
```
L^c_Grad-CAM = ReLU(Σ_k α^c_k A^k)

where α^c_k = (1/Z) Σ_i Σ_j (∂y^c / ∂A^k_{ij})
```

**Interpretation:** Highlights regions that contribute to class prediction.

### 2. Saliency Maps

**Reference:** Simonyan et al., "Deep Inside Convolutional Networks", 2013

**Formula:**
```
S(x) = |∇_x F_y(x)|
```

**Interpretation:** Pixel-wise gradient magnitude showing sensitivity.

### 3. Integrated Gradients

**Reference:** Sundararajan et al., "Axiomatic Attribution for Deep Networks", ICML 2017

**Formula:**
```
A(x) = (x - x_baseline) ⊙ ∫_0^1 ∇F(x_baseline + α(x - x_baseline)) dα
```

**Interpretation:** Attributions satisfying axioms of sensitivity and implementation invariance.

---

## Output Files

### Model Checkpoints
- `checkpoints/best_model.pth`: Best validation model
- `final_model.pth`: Final trained model

### Metrics
- `test_metrics.json`: Comprehensive evaluation metrics
- `train_history.json`: Training/validation curves

### Splits
- `splits/train.csv`: Training split
- `splits/val.csv`: Validation split
- `splits/test.csv`: Test split

### Visualizations
- `plots/training_history.png`: Loss and accuracy curves
- `plots/confusion_matrix.png`: Confusion matrix (counts & normalized)
- `plots/per_class_metrics.png`: Per-class precision, recall, F1

### XAI Outputs
- `xai/sample_*_gradcam_saliency.png`: Grad-CAM and saliency visualizations

---

## Evaluation Metrics Explained

### 1. Accuracy
Proportion of correctly classified samples.

### 2. F1-Score
Harmonic mean of precision and recall.
- **Macro**: Unweighted mean across classes
- **Micro**: Global calculation from aggregated counts
- **Weighted**: Weighted by support (number of true instances)

### 3. Expected Calibration Error (ECE)
Measures how well predicted probabilities match actual accuracy.
- Lower ECE = better calibrated model
- Important for safety-critical applications

### 4. Brier Score
Mean squared error of probability forecasts.
- Range: [0, 1]
- Lower is better
- Proper scoring rule

### 5. Attribution Sparsity
Fraction of pixels with low attribution values.
- Higher sparsity = more focused explanations
- Indicates model reliance on specific regions

---

## Statistical Significance Testing

To run multiple independent experiments:

```python
from prototypical_xai_complete import main, Config

# Run 3 independent experiments
results = main(CFG, run_experiments=True, n_experiment_runs=3)

# Results include:
# - Mean and std of accuracy across runs
# - Mean and std of F1-macro across runs
# - Individual run metrics
```

This provides confidence intervals for reported metrics.

---

## Model Architecture

### ConvEncoder

```
Input: [B, 3, 128, 128]
  ↓ Conv2d(3→32, 3×3) + BN + ReLU(inplace=False) + MaxPool(2) + Dropout
  ↓ Conv2d(32→64, 3×3) + BN + ReLU(inplace=False) + MaxPool(2) + Dropout
  ↓ Conv2d(64→128, 3×3) + BN + ReLU(inplace=False) + MaxPool(2) + Dropout
  ↓ Conv2d(128→256, 3×3) + BN + ReLU(inplace=False) + AdaptiveAvgPool
  ↓ Flatten
  ↓ FC(256→256) + ReLU(inplace=False) + Dropout
  ↓ FC(256→128)
  ↓ L2 Normalization
Output: [B, 128]
```

**Note:** All ReLU layers use `inplace=False` to ensure compatibility with GradCAM backward hooks.

### PrototypicalNetwork

```
Support Images: [N×K, 3, H, W] ─┐
                                ├→ Encoder → Prototypes: [N, D]
Query Images: [N×Q, 3, H, W] ───┤
                                └→ Encoder → Query Embeddings: [N×Q, D]

Distance Computation: d(query_i, prototype_j) = ‖z_qi - p_j‖²

Logits: -distances

Softmax → Class Probabilities
```

---

## Citations

If using this implementation in your research:

```bibtex
@inproceedings{snell2017prototypical,
  title={Prototypical networks for few-shot learning},
  author={Snell, Jake and Swersky, Kevin and Zemel, Richard},
  booktitle={Advances in neural information processing systems},
  year={2017}
}

@inproceedings{selvaraju2017grad,
  title={Grad-cam: Visual explanations from deep networks via gradient-based localization},
  author={Selvaraju, Ramprasaath R and others},
  booktitle={ICCV},
  year={2017}
}

@inproceedings{sundararajan2017axiomatic,
  title={Axiomatic attribution for deep networks},
  author={Sundararajan, Mukund and Taly, Ankur and Yan, Qiqi},
  booktitle={ICML},
  year={2017}
}
```

---

## Troubleshooting

### CUDA Out of Memory
- Reduce `img_size` (e.g., 128 → 96)
- Reduce `q_query` (e.g., 15 → 10)
- Reduce `episodes_per_epoch`

### Poor Performance
- Increase `n_epochs`
- Adjust learning rate
- Check data augmentation
- Verify data split stratification

### Slow Training
- Ensure GPU is being used: `torch.cuda.is_available()`
- Enable cuDNN benchmarking: `torch.backends.cudnn.benchmark = True`

### RuntimeError: BackwardHookFunctionBackward is a view and is being modified inplace

**Cause:** The `inplace=True` setting in ReLU layers conflicts with PyTorch backward hooks used in GradCAM.

**Fix:** All ReLU layers in `ConvEncoder` must use `inplace=False`:

```python
# Correct (for GradCAM compatibility)
nn.ReLU(inplace=False)

# Incorrect (causes error with backward hooks)
nn.ReLU(inplace=True)
```

This fix has been applied to all model implementations in this repository. If you modify the encoder architecture, ensure all ReLU layers use `inplace=False` to maintain GradCAM compatibility.

---

## License

This implementation is provided for research purposes. Please cite the original papers when using in publications.

---

## Contact

For questions or issues, please open an issue in the repository.

**Last Updated:** 2026-04-06 (Fixed inplace ReLU compatibility with GradCAM)
