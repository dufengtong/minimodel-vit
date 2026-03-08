# minimodel-vit

Encoding models for mouse primary visual cortex (V1) using a Vision Transformer (ViT) backbone with the [minimodel](https://github.com/MouseLand/minimodel) factorized readout. The project compares two regimes:

1. **Frozen backbone** — the ViT is used purely as a feature extractor; only the linear readout is trained.
2. **Fine-tuned backbone** — the last *N* transformer blocks are unfrozen and jointly optimized with the readout at a lower learning rate.

Results are compared against the CNN minimodel (2-layer depth-separable convolutional core + the same readout) to assess what large-scale visual pretraining adds to neural predictivity.

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

### ViT backbone (DINOv3 ViT-Small)

| Property | Value |
|---|---|
| Model | `facebook/dinov3-vits16-pretrain-lvd1689m` |
| Parameters | ~21 M |
| Patch size | 16 × 16 |
| Embedding dim | 384 |
| Transformer blocks | 12 |
| Spatial output grid | 4 × 8 (from 64 × 128 input) |
| Register tokens | 4 (DINOv3) |

Features can be extracted from any subset of the 12 transformer blocks, or from the initial patch-embedding convolution (block 0), giving 13 possible readout points in total.

### Readout

The readout is identical to the minimodel factorized readout:

- **Rank-1, yx-separable spatial weights** — each neuron learns a separable 2-D weight map `Wy ⊗ Wx` over the 4 × 8 patch grid, clamped to be non-negative during training.
- **Channel weights** `Wc` — a learned linear combination over the 384 feature channels (or projected dimension if `use_channel_proj=True`).
- **Output nonlinearity** — ELU + bias, followed by Poisson (softplus) output to enforce positivity.
- **Loss** — Poisson negative log-likelihood.

```
ViT features  (N, 384, 4, 8)
      ↓  Wc  (n_neurons, 384)
      ↓  Wy ⊗ Wx  (n_neurons, 4, 8)
      ↓  ELU + bias
      ↓  Poisson output
predicted firing rates  (N, n_neurons)
```

### Training protocol

**Frozen backbone:**
- Only the readout parameters (`Wc`, `Wy`, `Wx`, `bias`) are optimized.
- Optimizer: AdamW, `lr=1e-3`, L2 on `Wc` (`l2_readout=0.1`).
- Early stopping on validation variance explained (patience = 5 epochs, up to 200 epochs).

**Fine-tuned backbone:**
- Initialized from the frozen-backbone checkpoint.
- Last *N* transformer blocks unfrozen (default `N=2`).
- Two learning-rate groups: backbone `lr=1e-5`, readout `lr=1e-3`.
- L2 on `Wc` (`l2_readout=1e-4`).
- Early stopping (patience = 5, up to 100 epochs).

---

## Results

### FEVE by ViT block (frozen backbone)

FEVE (fraction of explainable variance explained) measured on 500 held-out test images with 10 repeats. Models are trained on mouse FX10 (`mouse_id=3`, 4 792 neurons). Valid neurons are defined as FEV > 0.15 (3 040 / 4 792).

Each point on the x-axis corresponds to features extracted from the output of that transformer block (block 0 = patch-embedding convolution, blocks 1–12 = transformer outputs).

<!-- INSERT FIGURE: FEVE vs. block index (bar or line plot, x=0..12, y=FEVE) -->

### Frozen vs. fine-tuned vs. CNN minimodel

<!-- INSERT FIGURE: scatter plot (frozen CC vs. fine-tuned CC, per neuron) -->

<!-- INSERT FIGURE: scatter plot (CNN minimodel CC vs. fine-tuned CC, per neuron) -->

<!-- INSERT FIGURE: histogram of ΔCC (fine-tuned − frozen) and ΔCC (fine-tuned − CNN) -->

#### Summary table

| Model | FEVE (FEV > 0.15) |
|---|---|
| CNN minimodel (2-layer, 16/320 filters) | — |
| ViT frozen (best block) | — |
| ViT fine-tuned (last 2 blocks) | — |

*(Fill in numbers after running experiments.)*

### Discussion

**Within-model block comparison.**
Early blocks (1–4) of DINOv3 ViT-Small capture low-to-mid-level features (edges, textures) most relevant for V1 neurons. Later blocks tend to encode higher-level semantic content that is less predictive of V1 responses. The patch-embedding layer (block 0) provides a simple linear projection of local patches and serves as a useful lower bound.

**Frozen vs. fine-tuned.**
Fine-tuning the last transformer blocks with a small backbone learning rate allows the model to adapt pre-trained features toward the statistics of natural images seen by the mouse, potentially recovering some of the spatial precision lost when applying ImageNet-pretrained representations to grayscale stimuli.

**ViT vs. CNN minimodel.**
The CNN minimodel uses a learned 2-layer convolutional core optimized end-to-end on the same neural data. Comparing it with the frozen ViT isolates the contribution of large-scale pretraining versus task-specific optimization. Comparing with the fine-tuned ViT reveals whether the 21 M-parameter pretrained backbone—even partially adapted—provides complementary representational structure that the small CNN cannot capture.

---

## Repository structure

```
minimodel-vit/
├── notebooks/
│   ├── vit_frozen_mouse.ipynb      # frozen-backbone training & evaluation (interactive)
│   └── vit_finetune_mouse.ipynb    # fine-tuning & three-way comparison (interactive)
├── scripts/
│   ├── vit_frozen_mouse.py         # training script (cluster / command-line)
│   ├── vit_frozen_mouse_train.py   # bsub job launcher (LSF cluster)
│   └── vit_frozen_mouse_script_a100.sh  # shell wrapper for each bsub job
└── utils/
    ├── data.py                     # data loading & preprocessing (from minimodel)
    ├── metrics.py                  # FEVE / FEV metrics (from minimodel)
    ├── model_builder.py            # CNN core + readout (from minimodel, for comparison)
    ├── model_trainer.py            # training loops (from minimodel)
    └── vit_encoding_model.py       # ViT backbone + readout (TODO: implement)
```

`utils/` contains code adapted from [MouseLand/minimodel](https://github.com/MouseLand/minimodel). `utils/vit_encoding_model.py` is a stub that documents the required API; it needs to be implemented before running any notebooks or scripts.

---

## Requirements

```
torch >= 2.0
torchvision
numpy
scipy
opencv-python
transformers        # HuggingFace, for DINOv3 weights
tqdm
matplotlib          # notebooks only
```

Install dependencies:

```bash
conda activate minimodel_env
pip install transformers tqdm
```

A HuggingFace token is required to download `facebook/dinov3-vits16-pretrain-lvd1689m`. Pass it via `--hf_token` or set it directly in the launcher script.

---

## Usage

### Notebooks (interactive)

Open `notebooks/vit_frozen_mouse.ipynb` or `notebooks/vit_finetune_mouse.ipynb` in Jupyter. Set the configuration cell at the top (mouse ID, HF token, layer config) and run all cells.

### Cluster (LSF / bsub)

```bash
cd scripts/
# Edit mouse_ids, extract_layers, hf_token in vit_frozen_mouse_train.py, then:
python vit_frozen_mouse_train.py
```

Each job calls `vit_frozen_mouse_script_a100.sh` which activates `minimodel_env` and runs `vit_frozen_mouse.py`.

### Command-line (single run)

```bash
cd scripts/
python vit_frozen_mouse.py \
    --mouse_id 3 \
    --extract_layers 4 \
    --hf_token YOUR_TOKEN \
    --data_path ../data \
    --weight_path ../checkpoints/vit_frozen
```
