# minimodel-vit

Encoding models for mouse primary visual cortex (V1) using a Vision Transformer (ViT) backbone with the [minimodel](https://github.com/MouseLand/minimodel) readout. The project asks a simple question: **can we directly plug frozen (or lightly fine-tuned) DINOv3 patch features into the minimodel readout and match the performance of the task-optimized CNN core?**

Two training regimes are compared:

1. **Frozen backbone** — the ViT is used purely as a feature extractor; only the linear readout is trained.
2. **Fine-tuned backbone** — the last *N* transformer blocks are unfrozen and jointly optimized with the readout at a lower learning rate (experiments: N = 1, 2, 3, 4).

Results are compared against the CNN fullmodel (2-layer depth-separable convolutional core + readout) to assess what large-scale visual pretraining adds to neural predictivity.

---

## Model structure

### Image preprocessing

Raw stimuli are 66 × 264 grayscale images displayed on a wide-field monitor. Before entering the ViT, each image is resized and cropped to match the receptive-field coverage of the recorded neurons:

```
66 × 264  →  resize  →  64 × 256
          →  crop (left half)  →  64 × 128
          →  per-image [0, 1] normalization
          →  replicate to 3 channels  (grayscale → pseudo-RGB)
          →  ImageNet normalize  (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

Output: `(N, 3, 64, 128)` tensors fed to the ViT.

### ViT backbone (DINOv3 ViT-Small / ViT-Base)

| Property | ViT-S/16 | ViT-B/16 |
|---|---|---|
| HuggingFace ID | `facebook/dinov3-vits16-pretrain-lvd1689m` | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| Parameters | ~21 M | ~86 M |
| Patch size | 16 × 16 | 16 × 16 |
| Embedding dim | 384 | 768 |
| Transformer blocks | 12 | 12 |
| Spatial output grid | 4 × 8 (from 64 × 128 input) | 4 × 8 |
| Register tokens | 4 (DINOv3) | 4 |

Features can be extracted from any of the 12 transformer blocks, giving 12 possible readout points in total.

### Architecture: how ViT and readout are concatenated

The overall pipeline is:

```
Input image (N, 3, 64, 128)
      ↓  resize to ViT input size
      ↓  ViT backbone (frozen or partially fine-tuned)
      ↓  extract patch tokens from block k  →  (N, num_patches, D)
      ↓  drop CLS + register tokens, reshape to spatial grid
patch features  (N, D, 4, 8)         ← D = 384 (ViT-S) or 768 (ViT-B)
      ↓  Wc  (n_neurons, 384/768)        ← channel weights (linear combination)
      ↓  Wy ⊗ Wx  (n_neurons, 4, 8) ← rank-1 separable spatial weights (≥ 0)
      ↓  ELU + bias
      ↓  + 1  (Poisson output shift)
predicted firing rates  (N, n_neurons)
```

The key design choice is that **each 16×16 patch token is treated as one spatial location** in a 4×8 grid. The readout from the minimodel learns a per-neuron separable weight map over this grid and a linear combination of feature channels. This is identical to the readout applied to the CNN core feature maps, making the comparison between ViT and CNN fair at the readout level.

### Training protocol

**Frozen backbone:**
- Only the readout parameters (`Wc`, `Wy`, `Wx`, `bias`) are optimized.
- Optimizer: AdamW, `lr=1e-3`, L2 on `Wc` (`l2_readout=0.1`).
- LR scheduler: ReduceLROnPlateau (halve when val varexp stalls, patience=5).
- Early stopping when LR reaches `min_lr=1e-5`.

**Fine-tuned backbone:**
- Initialized from the frozen-backbone checkpoint (warm-started readout).
- Last *N* transformer blocks unfrozen (experiments: N = 1, 2, 3, 4).
- Three learning-rate groups: backbone `lr=1e-5`, readout `lr=1e-3`.
- LR schedule: CosineAnnealingLR over max 100 epochs.
- Early stopping (patience = 5, up to 100 epochs).

---

## Results

All results are on mouse **FX10** (4,792 neurons, 500 held-out test images × 10 repeats). Valid neurons are defined as FEV > 0.15 (approximately 3,040 / 4,792 neurons pass). FEVE (fraction of explainable variance explained) is averaged over valid neurons.

### Figure 1a — FEVE by block (frozen ViT-S/16)

<div align="center"><img src="figures/fig1a_frozen_feve_vs_block.png" width="80%"></div>

Frozen ViT-S/16 features extracted from each transformer block. FEVE peaks at **block 4 (~0.32)**, indicating that low-to-mid-level representations are most predictive of V1 responses. Performance declines steadily in later blocks as representations become more semantic.

### Figure 1b — Frozen vs. Fine-tuned ViT-S/16

<div align="center"><img src="figures/fig1b_frozen_vs_ft2_feve_vs_block.png" width="80%"></div>

Fine-tuning the last 2 blocks consistently improves FEVE across all ViT depths. The fine-tuned peak rises to **block 3–4 (~0.52)**, up from ~0.32 frozen. The block-ordering of performance is preserved after fine-tuning.

### Figure 2 — FEVE distribution at best blocks (ViT-S/16)

<div align="center"><img src="figures/fig2_best_blocks_comparison.png" width="80%"></div>

Violin plots of per-neuron FEVE at blocks 3 and 4 across five conditions: Frozen, FT-1, FT-2, FT-3, and FT-4 blocks. Fine-tuning raises the mean FEVE progressively with each additional unfrozen block, from **~0.32 (frozen)** to **~0.54 (FT-4 blocks)**. 

### Figure 2b — Best-block FEVE distribution (ViT-S vs. ViT-B, all fine-tuning levels)

<div align="center"><img src="figures/fig2b_vits_vs_vitb_best_blocks.png" width="100%"></div>

Side-by-side comparison of per-neuron FEVE at the best block for ViT-S/16 and ViT-B/16 across all fine-tuning conditions. In the frozen setting, ViT-B/16 outperforms ViT-S/16 (0.38 vs. 0.32). After fine-tuning 4 blocks, **ViT-S/16 slightly surpasses ViT-B/16** (0.5449 vs. 0.5309), suggesting the smaller model adapts more efficiently.

### Figure 3 — ViT-S/16 vs. ViT-B/16 (frozen)

<div align="center"><img src="figures/fig3_vits_vs_vitb_frozen_feve_vs_block.png" width="80%"></div>

Comparing frozen ViT-Small (384-dim) and ViT-Base (768-dim). ViT-B/16 outperforms ViT-S/16 in early-to-mid blocks, peaking at **block 3 (~0.38 vs. ~0.32)**. Both models degrade sharply at later blocks.

### Figure 4 — All conditions (frozen & fine-tuned, ViT-S and ViT-B)

<div align="center"><img src="figures/fig4_vits_vs_vitb_frozen_and_ft_feve_vs_block.png" width="80%"></div>

Full comparison across both model sizes and both training regimes. All curves peak around blocks 3–4. Fine-tuning narrows the gap between ViT-S and ViT-B. Later blocks (9–11) show divergent behavior after fine-tuning, likely due to overfitting with less-predictive features.

### Figure 5 — Best-block FEVE (ViT-S vs. ViT-B, frozen and FT-2)

<div align="center"><img src="figures/fig5_best_block_vits_vs_vitb.png" width="80%"></div>

Mean FEVE at the best block, comparing frozen and FT-2-block conditions for both architectures. Fine-tuning 2 blocks roughly doubles the frozen-to-FT gain relative to the old runs, reaching ~0.51 for both ViT-S and ViT-B.

### Summary table

| Model | Best block | FEVE (mean, valid neurons) |
|---|---|---|
| CNN fullmodel (2-layer depth-sep. conv, 16/320 filters) | — | **0.6654** |
| ViT-S/16 frozen | 4 | 0.3234 |
| ViT-S/16 FT-1 block | 3 | 0.4896 |
| ViT-S/16 FT-2 blocks | 3 | 0.5168 |
| ViT-S/16 FT-3 blocks | 3 | 0.5255 |
| ViT-S/16 FT-4 blocks | 4 | **0.5449** (best ViT-S) |
| ViT-B/16 frozen | 3 | 0.3787 |
| ViT-B/16 FT-1 block | 3 | 0.4712 |
| ViT-B/16 FT-2 blocks | 3 | 0.5112 |
| ViT-B/16 FT-3 blocks | 3 | 0.5293 |
| ViT-B/16 FT-4 blocks | 3 | 0.5309 |

The best ViT model (ViT-S/16 FT-4 blocks) achieves **0.5449 FEVE**, compared to **0.6654 FEVE** for the CNN fullmodel.

---

## Discussion

### Why does the CNN model still outperform the ViT?

Despite the large improvement from fine-tuning (0.32 → 0.54), the CNN fullmodel (FEVE 0.6654) retains a ~0.12 FEVE advantage. Several factors likely contribute:

#### Spatial resolution and the patch-grid readout

Mouse V1 neurons have spatially localized receptive fields. The CNN fullmodel processes the image at full spatial resolution through convolutional layers, preserving fine-grained spatial structure. In contrast, the ViT backbone divides the 64×128 input into a coarse **4×8 grid of 16×16 patches**. By treating each patch token as a single spatial location in the readout, we limit spatial precision to 16-pixel granularity. If a neuron's RF is smaller than or misaligned with a patch boundary, the spatial weight map cannot capture it accurately. The CNN, with its strided convolutions and dense spatial feature maps, is better suited to this regime.

#### Self-attention mixes spatial information

In a transformer, every patch token attends to every other patch token. Even at early blocks (where FEVE peaks), each spatial location has already integrated global context through self-attention. This contrasts with V1 neurons, which pool from localized spatial regions.

### Block-ordering of performance

The consistent FEVE peak at blocks 3–4 (out of 12) for both ViT-S and ViT-B, across all fine-tuning levels, is a robust finding. It mirrors results from neural predictivity studies of ImageNet-pretrained ViTs in primate: mid-early layers best predict V1/V2 responses, while deeper layers are better aligned with higher visual areas (V4, IT).

### Fine-tuning

Fine-tuning consistently improves FEVE, with gains that accumulate up to 4 unfrozen blocks. The largest single jump is frozen → FT-1 (~0.17 FEVE for ViT-S/16 at block 3), with diminishing returns at FT-3 and FT-4. This suggests that the pretrained representations are substantially adaptable, but additional unfrozen blocks provide only marginal benefit beyond the first few.

### ViT-S vs. ViT-B

ViT-B/16 (86 M params) outperforms ViT-S/16 (21 M params) in the frozen setting at early-to-mid blocks (0.38 vs. 0.32), but the gap reverses slightly after fine-tuning 4 blocks (ViT-S 0.5449 vs. ViT-B 0.5309). This suggests that ViT-S may adapt more efficiently when fine-tuned on relatively small neural datasets comparing to its large-scale pretraining dataset (LVD-142M).

---

## Requirements

```
torch >= 2.0
torchvision
numpy
scipy
opencv-python
transformers
tqdm
matplotlib
```
