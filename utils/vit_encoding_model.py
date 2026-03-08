"""
ViT Encoding Model for Neural Responses
========================================
A clean ViT-based encoding model that reuses the same Readout from model_builder.

Changes from v1:
  - token_type: 'patch' (spatial, default) or 'cls' (global CLS token)
  - extract_layers: which transformer blocks to pull features from;
    reflected in make_model_name() for easy checkpoint identification
  - Training: patience-based early stopping + ReduceLROnPlateau (no fixed-period schedule)
  - Image preprocessing: handle new 32x64 model input
    (data pipeline: 66x264 → resize 64x256 → crop 64x128 → downsample 32x64)

Recommended model for RTX 3090 fine-tuning:
    'facebook/dinov3-vits16-pretrain-lvd1689m'  # ViT-Small, 384-dim, ~21M params
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from minimodel.model_builder import Readout

HF_TOKEN = ""

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ============================================================
# Utilities
# ============================================================

def _model_shortname(model_name):
    """Extract a compact variant name from a HuggingFace model ID.

    Examples:
        'facebook/dinov3-vits16-pretrain-lvd1689m' → 'dinov3_vits16'
        'facebook/dinov2-vitb14'                   → 'dinov2_vitb14'
    """
    basename = model_name.split('/')[-1]
    for prefix in ('dinov3-', 'dinov2-'):
        if prefix in basename:
            variant = basename.split(prefix)[1].split('-')[0]
            return f"{prefix.rstrip('-')}_{variant}"
    # Fallback: first two dash-separated tokens
    parts = basename.split('-')
    return '_'.join(parts[:2])


def make_model_name(mouse_name, exp_date, model_name, token_type, extract_layers):
    """
    Build a descriptive checkpoint filename.

    Format: {mouse}_{date}_{model}_{token}_{layers}.pt

    Examples:
        'FX10_051623_dinov3_vits16_patch_last.pt'
        'FX10_051623_dinov3_vits16_cls_l8-11.pt'

    Args:
        mouse_name:     e.g. 'FX10'
        exp_date:       e.g. '051623'
        model_name:     HuggingFace model ID
        token_type:     'patch' or 'cls'
        extract_layers: None (last layer) or list of ints, e.g. [8, 9, 10, 11]
    """
    short = _model_shortname(model_name)
    if extract_layers is None:
        layer_str = 'last'
    elif len(extract_layers) == 1:
        layer_str = f'l{extract_layers[0]}'
    else:
        layer_str = f'l{extract_layers[0]}-{extract_layers[-1]}'
    return f'{mouse_name}_{exp_date}_{short}_{token_type}_{layer_str}.pt'


# ============================================================
# 1. ViT Core
# ============================================================

class ViTCore(nn.Module):
    """
    DINOv3 ViT feature extractor with GPU-side preprocessing.

    Takes (B, 3, H, W) ImageNet-normalized tensors (output of preprocess_images()).
    Outputs (B, C_out, Ly_out, Lx_out) spatial feature maps.

    Pipeline:
      1. Preprocess on GPU: 1-ch → 3-ch, [0,1] rescale, resize, ImageNet normalize.
      2. Run ViT → extract token features.
         - 'patch': patch tokens → reshape to (B, D, H_p, W_p).
         - 'cls':   CLS token   → reshape to (B, D, 1, 1).
      3. Optional 1x1 conv projection: D → proj_dim.
      4. Optional bilinear upsample to out_spatial_size (patch only).

    Args:
        model_name: HuggingFace model ID.
            RTX 3090 recommended → 'facebook/dinov3-vits16-pretrain-lvd1689m'
        vit_input_size: (H, W) to resize images to; must be divisible by patch_size (16).
            (112, 224) gives 7x14 patches; aspect ratio ≈ 0.5 ≈ 32x64 input aspect.
        out_spatial_size: (Ly, Lx) to upsample patch features to for the readout.
            (16, 32) matches 2x downsampling of the 32x64 model input.
            None = keep native patch grid.
            Ignored when token_type='cls' (CLS is already 1x1).
        extract_layers: list of transformer block indices for multi-layer features.
            None = last layer only.
        token_type: 'patch' (spatial readout) or 'cls' (global linear readout).
        freeze: freeze all ViT parameters.
        use_channel_proj: 1x1 Conv-BN-ReLU to project to proj_dim channels.
        proj_dim: output channels after projection.
        hf_token: HuggingFace token for private/gated models.
    """

    def __init__(self, model_name,
                 vit_input_size=(64, 128),
                 out_spatial_size=None,
                 extract_layers=None,
                 freeze=True,
                 use_channel_proj=True, proj_dim=64,
                 hf_token=""):
        super().__init__()

        # -- Backbone --
        self.backbone = AutoModel.from_pretrained(
            model_name, token=hf_token if hf_token else None
        )
        cfg = self.backbone.config
        self.patch_size           = cfg.patch_size
        self.embed_dim            = cfg.hidden_size
        self.num_register_tokens  = getattr(cfg, 'num_register_tokens', 0)
        self.extract_layers       = extract_layers
        self.freeze               = freeze
        self.vit_input_size       = vit_input_size

        # CLS has no spatial structure; ignore out_spatial_size
        self.out_spatial_size = out_spatial_size 

        # Patch grid dimensions (used for patch token reshaping)
        self.H_patches = vit_input_size[0] // self.patch_size
        self.W_patches = vit_input_size[1] // self.patch_size

        # Raw channel count before projection
        n_layers     = len(extract_layers) if extract_layers is not None else 1
        raw_channels = self.embed_dim * n_layers

        # Optional 1x1 conv projection (works for both patch and CLS via spatial 1x1)
        self.use_channel_proj = use_channel_proj
        if use_channel_proj:
            self.channel_proj = nn.Sequential(
                nn.Conv2d(raw_channels, proj_dim, 1, bias=False),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True),
            )
            self.out_channels = proj_dim
        else:
            self.out_channels = raw_channels

        if freeze:
            self._freeze_backbone()

        spatial_info = f"{self.H_patches}x{self.W_patches}"
        print(f"ViTCore: {model_name}")
        print(f"  patch_size={self.patch_size} "
              f"embed_dim={self.embed_dim}, register_tokens={self.num_register_tokens}")
        print(f"  ViT input: {vit_input_size}  →  spatial output: {spatial_info}")
        print(f"  out_channels={self.out_channels}, out_spatial_size={self.out_spatial_size}")
        print(f"  frozen={freeze}")

    # ------------------------------------------------------------------
    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        self.freeze = True

    def unfreeze_last_n_blocks(self, n):
        """Unfreeze the last n transformer blocks + final layernorm for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        encoder_layers = self.backbone.encoder.layer
        total = len(encoder_layers)
        n = min(n, total)

        for layer in encoder_layers[total - n:]:
            for param in layer.parameters():
                param.requires_grad = True

        if hasattr(self.backbone, 'layernorm'):
            for param in self.backbone.layernorm.parameters():
                param.requires_grad = True

        self.freeze = False
        n_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"Unfroze last {n}/{total} ViT blocks ({n_trainable:,} backbone params trainable)")

    # ------------------------------------------------------------------
    def preprocess(self, img):
        """
        Resize (B, 3, H, W) ImageNet-normalized images to ViT input size.

        preprocess_images() already handles [0,1] scaling, 3-channel repeat,
        and ImageNet normalization — so this only needs to resize.
        """
        return F.interpolate(img, size=self.vit_input_size, mode='bilinear', align_corners=False)

    def _extract_features(self, pixel_values):
        """Run ViT and return (B, D[*n_layers], H_out, W_out)."""
        skip = 1 + self.num_register_tokens   # CLS + register tokens to skip

        if self.extract_layers is not None:
            # Multi-layer extraction
            outputs = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
            parts = []
            for idx in self.extract_layers:
                # hidden_states[0] = embedding; hidden_states[i+1] = block i
                hidden = outputs.hidden_states[idx + 1]
                patch = hidden[:, skip:, :]                      # (B, N, D)
                B, N, D = patch.shape
                parts.append(patch.transpose(1, 2).reshape(
                    B, D, self.H_patches, self.W_patches))

            return torch.cat(parts, dim=1)                       # (B, D*n_layers, H_p, W_p)
        else:
            # Last layer only
            outputs = self.backbone(pixel_values=pixel_values)
            patch = outputs.last_hidden_state[:, skip:, :]       # (B, N, D)
            B, N, D = patch.shape
            return patch.transpose(1, 2).reshape(
                B, D, self.H_patches, self.W_patches)            # (B, D, H_p, W_p)

    # ------------------------------------------------------------------
    def forward(self, img):
        """
        Args:
            img: (B, 3, H, W) ImageNet-normalized tensors on device.
        Returns:
            features: (B, C_out, Ly, Lx) spatial feature map.
                      Ly=Lx=1 when token_type='cls'.
        """
        pixel_values = self.preprocess(img)

        if self.freeze:
            with torch.no_grad():
                features = self._extract_features(pixel_values)
        else:
            features = self._extract_features(pixel_values)

        if self.use_channel_proj:
            features = self.channel_proj(features)

        return features


# ============================================================
# 2. ViT Encoder (core + readout)
# ============================================================

class ViTEncoder(nn.Module):
    """
    Full encoding model: ViTCore + Readout.
    Interface is identical to model_builder.Encoder.
    """

    def __init__(self, core, readout, loss_fun='poisson'):
        super().__init__()
        self.core      = core
        self.readout   = readout
        self.loss_fun  = loss_fun
        self.bias      = 1e-12

    def forward(self, img, detach_core=False):
        x = self.core(img)
        # print(f"Core output shape: {x.shape} | min/max: {x.min().item():.4f}/{x.max().item():.4f}")
        if detach_core:
            x = x.detach()
        x = self.readout(x)
        # print(f"Readout output shape: {x.shape} | min/max: {x.min().item():.4f}/{x.max().item():.4f}")
        x += 1 + self.bias
        return x

    def loss_function(self, spks_batch, spks_pred,
                      l1_readout=0.0, l2_readout=0.0, hs_reg=0.0):
        if self.loss_fun == 'poisson':
            loss = (spks_pred - spks_batch * torch.log(spks_pred)).sum(axis=0)
        else:
            loss = ((spks_pred - spks_batch) ** 2).sum(axis=0)
        # loss += l1_readout * self.readout.l1_norm()
        loss += l2_readout * self.readout.l2_norm()
        # loss += hs_reg     * self.readout.hoyer_square()
        return loss.mean()


# ============================================================
# 3. Build function
# ============================================================

def build_vit_encoder(n_neurons, model_name,
                      vit_input_size=(112, 224),
                      out_spatial_size=(16, 32),
                      extract_layers=None,
                      use_channel_proj=True, proj_dim=64,
                      freeze_backbone=True,
                      poisson=True, Wc_coef=0.01,
                      hf_token="",
                      device=torch.device('cuda')):
    """
    Build a ViT-based neural encoding model.

    Args:
        n_neurons: number of neurons to predict.
        model_name: HuggingFace model ID.
            RTX 3090 → 'facebook/dinov3-vits16-pretrain-lvd1689m' (ViT-S, ~21M params)
        vit_input_size: (H, W) to resize images to inside ViT; divisible by 16.
            (64, 128): aspect ratio 0.5. Gives 4x8 patches.
        out_spatial_size: (Ly, Lx) for readout feature map.
            (32, 64): 2x downsampled from model input (matches CNN convention).
            None: use native ViT patch grid (4x8 for vit_input_size=(64, 128)).
            Ignored for token_type='cls'.
        extract_layers: which transformer blocks to use. None = last only.
        token_type: 'patch' → spatial readout; 'cls' → effective linear readout (Ly=Lx=1).
        use_channel_proj: 1x1 Conv to project embed_dim → proj_dim.
        proj_dim: channels after projection.
        freeze_backbone: freeze ViT (train readout + proj only).
        poisson: Poisson loss (True) or MSE (False).
        Wc_coef: readout Wc init scale.
        hf_token: HuggingFace token.
        device: torch.device.

    Returns:
        model: ViTEncoder on device.
    """
    core = ViTCore(
        model_name=model_name,
        vit_input_size=vit_input_size,
        out_spatial_size=out_spatial_size,
        extract_layers=extract_layers,
        freeze=freeze_backbone,
        use_channel_proj=use_channel_proj,
        proj_dim=proj_dim,
        hf_token=hf_token,
    )

    # Readout spatial dimensions
    # if token_type == 'cls':
    #     readout_Ly, readout_Lx = 1, 1
    if out_spatial_size is not None:
        readout_Ly, readout_Lx = out_spatial_size
    else:
        readout_Ly, readout_Lx = core.H_patches * core.patch_size, core.W_patches * core.patch_size

    in_shape = (core.out_channels, readout_Ly, readout_Lx)
    print(f"Readout in_shape (C, Ly, Lx): {in_shape}")

    readout = Readout(
        in_shape, n_neurons, rank=1, yx_separable=True,
        bias_init=None, poisson=poisson, Wc_coef=Wc_coef,
    )

    model = ViTEncoder(core, readout, loss_fun='poisson' if poisson else 'mse')
    model = model.to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total:,} | Trainable: {trainable:,} | Frozen: {total - trainable:,}")

    return model


# ============================================================
# 4. Image preprocessing helper
# ============================================================

def preprocess_images(img_np, batch_size=2000):
    """
    Preprocess raw 66x264 grayscale images to DINOv3-ready (N, 3, 32, 64) tensors.

    Pipeline:
      1. Resize  66x264  →  64x256  (bilinear)
      2. Crop left half  →  64x128
      3. Downsample 2x   →  32x64   (bilinear)
      4. Per-image [0, 1] normalization
      5. Repeat to 3 channels (grayscale → pseudo-RGB)
      6. ImageNet normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

    Args:
        img_np: numpy array (N, 66, 264) float32, raw pixel values (not z-scored).
        batch_size: images per batch (CPU; adjust if memory is tight).

    Returns:
        numpy array (N, 3, 32, 64) float32, ready to feed directly into ViTCore.
    """
    n   = img_np.shape[0]
    out = np.zeros((n, 64, 128), dtype=np.float32)

    for i in range(0, n, batch_size):
        batch = torch.from_numpy(img_np[i:i + batch_size]).unsqueeze(1).float()
        batch = F.interpolate(batch, size=(64, 256), mode='bilinear', align_corners=False)
        batch = batch[:, :, :, :128]                      # → (B, 1, 64, 128)
        # batch = F.interpolate(batch, size=(32, 64),  mode='bilinear', align_corners=False)
        out[i:i + batch_size] = batch.squeeze(1).numpy()

    # Per-image [0, 1] normalization
    b_min = out.min(axis=(1, 2), keepdims=True)
    b_max = out.max(axis=(1, 2), keepdims=True)
    out = (out - b_min) / np.maximum(b_max - b_min, 1e-8)

    # Grayscale → pseudo-RGB: (N, 32, 64) → (N, 3, 32, 64)
    out = np.stack([out, out, out], axis=1)

    # ImageNet normalization
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std  = np.array(IMAGENET_STD,  dtype=np.float32).reshape(1, 3, 1, 1)
    out = (out - mean) / std

    return out


# ============================================================
# 5. Training helpers
# ============================================================

def _val_epoch(model, img_val, spks_val, batch_size=64,
               device=torch.device('cuda'), l2_readout=0.0):
    """Validation pass. Returns (val_loss, varexp_per_neuron, predictions_cpu)."""
    n_val, n_neurons = spks_val.shape
    spks_gpu = spks_val.to(device)
    pred_all = torch.zeros((n_val, n_neurons), device=device)
    val_loss = 0.0
    n_batches = int(np.ceil(n_val / batch_size))

    model.eval()
    with torch.no_grad():
        for b in range(n_batches):
            s = b * batch_size
            e = min(n_val, s + batch_size)
            pred = model(img_val[s:e])
            val_loss += model.loss_function(spks_gpu[s:e], pred, l2_readout=l2_readout).item()
            pred_all[s:e] = pred

    val_loss /= n_batches

    residual = ((spks_gpu - pred_all) ** 2).sum(0)
    centered = spks_gpu - spks_gpu.mean(0)
    total    = (centered ** 2).sum(0).clamp(min=1e-8)
    varexp   = (1 - residual / total).cpu()

    return val_loss, varexp, pred_all.cpu()

def check_named_params(module, tag=""):
    for name, p in module.named_parameters():
        if p.requires_grad and not torch.isfinite(p).all():
            print(f"[BAD PARAM] {tag} {name} has NaN/Inf")
            return False
    return True

def check_named_grads(module, tag=""):
    for name, p in module.named_parameters():
        if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all():
            print(f"[BAD GRAD] {tag} {name} grad has NaN/Inf")
            return False
    return True

def train_readout(model, spks_train, spks_val, img_train, img_val,
                  max_epochs=200, batch_size=64, lr=1e-3,
                  l2_readout=0.1, clamp=True,
                  patience=5,
                  device=torch.device('cuda')):
    """
    Train readout (+ channel projection) with frozen ViT backbone.

    Uses ReduceLROnPlateau + patience-based early stopping.

    Args:
        model: ViTEncoder with frozen backbone.
        spks_train, spks_val: (N, n_neurons) torch tensors.
        img_train, img_val:   (N, 1, H, W)  torch tensors on device.
        max_epochs: hard upper limit on training epochs.
        lr: initial learning rate for AdamW.
        l2_readout: L2 weight decay on readout Wc.
        clamp: clamp Wx, Wy ≥ 0 after each step.
        patience: epochs without varexp improvement before early stop.
                  LR is reduced after patience//2 epochs without improvement.

    Returns:
        best_state_dict (OrderedDict, CPU tensors).
    """
    n_train = img_train.shape[0]

    param_groups = [
        {'params': [model.readout.Wy, model.readout.Wx], 'weight_decay': 1.0},
        {'params': [model.readout.Wc],                   'weight_decay': l2_readout},
        {'params': [model.readout.bias],                  'weight_decay': 0.0},
    ]
    if model.core.use_channel_proj:
        param_groups.append(
            {'params': list(model.core.channel_proj.parameters()), 'weight_decay': 0.1}
        )

    optimizer = torch.optim.AdamW(param_groups, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max',
        patience=max(1, patience),
        factor=0.5, min_lr=1e-5,
    )

    best_state        = None
    varexp_max        = -np.inf
    tic               = time.time()

    print(f"Training readout | max_epochs={max_epochs}, patience={patience}, lr={lr:.1e}")

    for epoch in range(max_epochs):
        # -- Train --
        model.train()
        if model.core.freeze:
            model.core.backbone.eval()   # keep BN fixed during frozen stage

        perm       = np.random.permutation(n_train)
        train_loss = 0.0
        n_batches  = int(np.ceil(n_train / batch_size))

        for b in range(n_batches):
            inds       = perm[b * batch_size: min(n_train, (b + 1) * batch_size)]
            spks_batch = spks_train[inds].to(device)
            pred       = model(img_train[inds])
            # print(f"Batch {b}/{n_batches} | pred shape: {pred.shape} | pred min/max: {pred.min().item():.4f}/{pred.max().item():.4f}")
            loss       = model.loss_function(spks_batch, pred, l2_readout=l2_readout)
            # print(f"Epoch {epoch} | Batch {b}/{n_batches} | Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()

            # check_named_grads(model.readout, f"epoch {epoch} batch {b} after backward")
            for name, p in model.readout.named_parameters():
                if p.grad is not None:
                    g = p.grad
                    if not torch.isfinite(g).all():
                        print(f"[BAD GRAD] {name}")
                        print("  nan:", torch.isnan(g).any().item())
                        print("  inf:", torch.isinf(g).any().item())
                        print("  grad abs max:", torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).abs().max().item())
                        break
                    if not torch.isfinite(p).all():
                        print(f"[BAD PARAM] {name}")
                        print("  nan:", torch.isnan(p).any().item())
                        print("  inf:", torch.isinf(p).any().item())
                        print("  param abs max:", torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0).abs().max().item())
                        break

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )

            # check_named_params(model.readout, f"epoch {epoch} batch {b} before step")
            optimizer.step()
            # check_named_params(model.readout, f"epoch {epoch} batch {b} after step")

            if clamp:
                model.readout.Wx.data.clamp_(0)
                model.readout.Wy.data.clamp_(0)

            train_loss += loss.item()
        train_loss /= n_train

        # -- Validate --
        val_loss, varexp, _ = _val_epoch(
            model, img_val, spks_val, batch_size=batch_size,
            device=device, l2_readout=l2_readout,
        )
        if not np.isnan(varexp.mean()):
            scheduler.step(varexp.mean())   # LR based on validation varexp
        else:
            print("nan varexp, skipping LR scheduler step")

        if varexp.mean() > varexp_max and not np.isnan(val_loss * train_loss):
            best_state        = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            varexp_max        = varexp.mean()
            epochs_since_best = 0
        elif np.isnan(val_loss * train_loss):
            print("nan loss, stopping")
            break

        lr_now = optimizer.param_groups[0]['lr']
        if epoch % 1 == 0 or epoch + 1 == max_epochs:
            print(f"  epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | "
                  f"varexp {varexp.mean():.4f} | lr {lr_now:.1e} | "
                  f"patience {epochs_since_best}/{patience} | {time.time()-tic:.1f}s")

        if lr_now <= scheduler.min_lrs[0]:
            print("LR reached min_lr, stopping training")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    return best_state


def finetune_vit(model, spks_train, spks_val, img_train, img_val,
                 n_blocks_to_unfreeze=2,
                 max_epochs=100, batch_size=32,
                 backbone_lr=1e-5, readout_lr=1e-3, l2_readout=1e-4,
                 clamp=True, patience=5,
                 device=torch.device('cuda')):
    """
    Fine-tune the last n transformer blocks with differential learning rates.
    Call AFTER train_readout() for a warm-started readout.

    RTX 3090 + ViT-S/16 recommended settings:
        n_blocks_to_unfreeze=2, batch_size=32, backbone_lr=1e-5

    Args:
        patience: epochs without varexp improvement before early stop.
    """
    model = model.to(device)
    model.core.unfreeze_last_n_blocks(n_blocks_to_unfreeze)

    backbone_params, proj_params, readout_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'readout' in name:
            readout_params.append(param)
        elif 'channel_proj' in name:
            proj_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr,    'weight_decay': 1e-2},
        {'params': proj_params,     'lr': backbone_lr * 5,'weight_decay': 1e-2},
        {'params': readout_params,  'lr': readout_lr,     'weight_decay': 1e-5},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-7
    )

    print(f"Fine-tuning | max_epochs={max_epochs}, patience={patience}")
    print(f"  Backbone: {sum(p.numel() for p in backbone_params):,} @ lr={backbone_lr:.1e}")
    print(f"  Proj:     {sum(p.numel() for p in proj_params):,} @ lr={backbone_lr*5:.1e}")
    print(f"  Readout:  {sum(p.numel() for p in readout_params):,} @ lr={readout_lr:.1e}")

    n_train           = img_train.shape[0]
    varexp_max        = -np.inf
    best_state        = None
    epochs_since_best = 0
    tic               = time.time()

    for epoch in range(max_epochs):
        model.train()
        perm       = np.random.permutation(n_train)
        train_loss = 0.0
        n_batches  = int(np.ceil(n_train / batch_size))

        for b in range(n_batches):
            inds = perm[b * batch_size: min(n_train, (b + 1) * batch_size)]
            pred = model(img_train[inds])
            loss = model.loss_function(spks_train[inds].to(device), pred, l2_readout=l2_readout)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

            if clamp:
                model.readout.Wx.data.clamp_(0)
                model.readout.Wy.data.clamp_(0)

            train_loss += loss.item()
        train_loss /= n_train
        scheduler.step()

        val_loss, varexp, _ = _val_epoch(
            model, img_val, spks_val, batch_size=batch_size,
            device=device, l2_readout=l2_readout,
        )

        if varexp.mean() > varexp_max and not np.isnan(val_loss * train_loss):
            varexp_max        = varexp.mean()
            best_state        = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if epoch % 5 == 0 or epoch + 1 == max_epochs:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} | "
                  f"varexp {varexp.mean():.4f} | lr {lr_now:.1e} | "
                  f"patience {epochs_since_best}/{patience} | {time.time()-tic:.1f}s")

        if epochs_since_best >= patience:
            print(f"Early stop at epoch {epoch} (best varexp={varexp_max:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    return best_state
