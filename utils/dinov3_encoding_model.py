"""
DINOv3 Encoding Model for Mouse V1 Neural Responses
=====================================================
Uses DINOv3 (from HuggingFace Transformers) as a frozen/fine-tunable 
feature extractor with a spatial readout adapted from the CNN minimodel.

DINOv3 key properties:
- Patch size: 16 (images should be divisible by 16)
- Uses RoPE positional embeddings (better resolution flexibility than DINOv2)
- Output: (B, 1 + num_register_tokens + num_patches, hidden_size)
  - Token 0: CLS token
  - Tokens 1..num_register_tokens: register tokens (global memory slots)
  - Remaining: patch tokens (spatial features)
- Available ViT variants: vits16 (384-dim), vitb16 (768-dim), vitl16 (1024-dim), etc.
- Also has ConvNext variants (outputs spatial features natively)

Author: Farah Du
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor


# ============================================================
# 1. DINOv3 ViT Feature Extractor (Core replacement)
# ============================================================

class DINOv3Core(nn.Module):
    """
    Wraps a DINOv3 ViT as a feature extractor that outputs
    spatially-arranged patch features: (B, D, H_patch, W_patch).
    
    This makes it compatible with the existing Readout class which
    expects (B, C, H, W) tensors for spatial einsum operations.
    
    Args:
        model_name: HuggingFace model name, e.g.:
            'facebook/dinov3-vits16-pretrain-lvd1689m' (384-dim, fastest)
            'facebook/dinov3-vitb16-pretrain-lvd1689m' (768-dim)
            'facebook/dinov3-vitl16-pretrain-lvd1689m' (1024-dim)
        input_size: (H, W) of images after preprocessing. Must be divisible by 16.
        extract_layers: list of layer indices to extract intermediate features from.
            If None, only use the last layer's patch tokens.
            If provided, concatenate features from all specified layers -> (B, D*len, H, W).
        freeze: if True, freeze all ViT parameters (for readout-only training).
        use_channel_proj: if True, add a 1x1 conv to project feature dim down.
        proj_dim: output channels if use_channel_proj is True.
    """
    def __init__(self, model_name='facebook/dinov3-vits16-pretrain-lvd1689m',
                 input_size=(224, 224),
                 extract_layers=None, freeze=True,
                 use_channel_proj=False, proj_dim=64):
        super().__init__()
        
        # Load pretrained DINOv3 ViT
        self.backbone = AutoModel.from_pretrained(model_name)
        self.config = self.backbone.config
        self.patch_size = self.config.patch_size  # 16 for DINOv3
        self.embed_dim = self.config.hidden_size
        self.num_register_tokens = self.config.num_register_tokens  # typically 4
        self.extract_layers = extract_layers
        self.freeze = freeze
        
        # Compute spatial output size
        self.H_patches = input_size[0] // self.patch_size
        self.W_patches = input_size[1] // self.patch_size
        self.num_patches = self.H_patches * self.W_patches
        
        # Determine output channel dim
        if extract_layers is not None:
            self.out_channels = self.embed_dim * len(extract_layers)
        else:
            self.out_channels = self.embed_dim
        
        # Optional channel projection (reduce dimensionality for tractable readout)
        self.use_channel_proj = use_channel_proj
        if use_channel_proj:
            self.channel_proj = nn.Sequential(
                nn.Conv2d(self.out_channels, proj_dim, 1, bias=False),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True)
            )
            self.out_channels = proj_dim
        
        # Freeze backbone if requested
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        
        print(f"DINOv3 Core initialized:")
        print(f"  Model: {model_name}")
        print(f"  Patch size: {self.patch_size}")
        print(f"  Embed dim: {self.embed_dim}")
        print(f"  Register tokens: {self.num_register_tokens}")
        print(f"  Spatial output: {self.H_patches} x {self.W_patches}")
        print(f"  Output channels: {self.out_channels}")
        print(f"  Frozen: {freeze}")
    
    def _extract_patch_features(self, pixel_values):
        """
        Extract patch token features from DINOv3 and reshape to spatial grid.
        
        DINOv3 output.last_hidden_state shape: (B, 1 + num_register + num_patches, D)
          - index 0: CLS token
          - index 1..num_register: register tokens
          - index num_register+1..: patch tokens
        """
        if self.extract_layers is not None:
            # Get intermediate hidden states
            outputs = self.backbone(
                pixel_values=pixel_values,
                output_hidden_states=True
            )
            # outputs.hidden_states is a tuple: (embed_out, layer0_out, layer1_out, ...)
            # Layer index i corresponds to hidden_states[i+1]
            features_list = []
            for layer_idx in self.extract_layers:
                hidden = outputs.hidden_states[layer_idx + 1]  # +1 because index 0 is embedding output
                # Extract patch tokens (skip CLS + register tokens)
                patch_tokens = hidden[:, 1 + self.num_register_tokens:, :]  # (B, num_patches, D)
                B, N, D = patch_tokens.shape
                spatial = patch_tokens.transpose(1, 2).reshape(B, D, self.H_patches, self.W_patches)
                features_list.append(spatial)
            features = torch.cat(features_list, dim=1)  # (B, D*n_layers, H, W)
        else:
            # Just use last layer
            outputs = self.backbone(pixel_values=pixel_values)
            last_hidden = outputs.last_hidden_state  # (B, 1 + reg + patches, D)
            # Extract patch tokens
            patch_tokens = last_hidden[:, 1 + self.num_register_tokens:, :]
            B, N, D = patch_tokens.shape
            features = patch_tokens.transpose(1, 2).reshape(B, D, self.H_patches, self.W_patches)
        
        return features
    
    def forward(self, pixel_values):
        """
        Args:
            pixel_values: (B, 3, H, W) preprocessed images (from DINOv3 image processor)
        Returns:
            features: (B, C_out, H_patch, W_patch) spatial feature map
        """
        if self.freeze:
            with torch.no_grad():
                features = self._extract_patch_features(pixel_values)
                features = features.detach()  # ensure no gradient flows back
        else:
            features = self._extract_patch_features(pixel_values)
        
        if self.use_channel_proj:
            features = self.channel_proj(features)
        
        return features
    
    def unfreeze_last_n_blocks(self, n):
        """Unfreeze the last n transformer blocks for fine-tuning."""
        # First, freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # DINOv3 ViT structure: backbone.encoder.layer[i]
        encoder_layers = self.backbone.encoder.layer
        total_layers = len(encoder_layers)
        
        for layer in encoder_layers[total_layers - n:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # Also unfreeze the final layernorm
        if hasattr(self.backbone, 'layernorm'):
            for param in self.backbone.layernorm.parameters():
                param.requires_grad = True
        
        self.freeze = False
        n_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"Unfroze last {n}/{total_layers} transformer blocks")
        print(f"  Backbone trainable params: {n_trainable:,}")
    
    def get_cls_token(self, pixel_values):
        """Get CLS token embedding (useful for linear probe baseline)."""
        with torch.no_grad():
            outputs = self.backbone(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0, :]  # (B, D)


# ============================================================
# 2. DINOv3 ConvNext Core (alternative - natively spatial)
# ============================================================

class DINOv3ConvNextCore(nn.Module):
    """
    Wraps DINOv3 ConvNext as feature extractor.
    ConvNext already outputs spatial features (B, C, H, W) natively,
    so no reshaping needed. Good alternative if you want CNN-like features
    from the DINOv3 family.
    
    Available models:
        'facebook/dinov3-convnext-tiny-pretrain-lvd1689m'
        'facebook/dinov3-convnext-small-pretrain-lvd1689m'
        'facebook/dinov3-convnext-base-pretrain-lvd1689m'
        'facebook/dinov3-convnext-large-pretrain-lvd1689m'
    """
    def __init__(self, model_name='facebook/dinov3-convnext-tiny-pretrain-lvd1689m',
                 freeze=True, use_channel_proj=False, proj_dim=64):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.freeze = freeze
        
        # ConvNext hidden_sizes: e.g., [96, 192, 384, 768] for tiny
        self.out_channels = self.backbone.config.hidden_sizes[-1]
        
        self.use_channel_proj = use_channel_proj
        if use_channel_proj:
            self.channel_proj = nn.Sequential(
                nn.Conv2d(self.out_channels, proj_dim, 1, bias=False),
                nn.BatchNorm2d(proj_dim),
                nn.ReLU(inplace=True)
            )
            self.out_channels = proj_dim
        
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
        
        print(f"DINOv3 ConvNext Core: {model_name}")
        print(f"  Output channels: {self.out_channels}")
    
    def forward(self, pixel_values):
        if self.freeze:
            with torch.no_grad():
                outputs = self.backbone(pixel_values=pixel_values)
                features = outputs.last_hidden_state.detach()  # (B, C, H, W)
        else:
            outputs = self.backbone(pixel_values=pixel_values)
            features = outputs.last_hidden_state
        
        if self.use_channel_proj:
            features = self.channel_proj(features)
        return features


# ============================================================
# 3. Readout (from your original code, unchanged)
# ============================================================

class Readout(nn.Module):
    """Spatial readout with separable spatial weights."""
    def __init__(self, in_shape, n_neurons, y_init=None, x_init=None, 
                 c_init=None, coef_init=None, rank=1, yx_separable=True, 
                 bias_init=None, poisson=False, activation='elu', 
                 Wc_coef=0.01, Wxy_gabor_init=0.1, Wxy_init=0.01):
        super().__init__()
        self.yx_separable = yx_separable
        n_conv, Ly, Lx = in_shape
        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        if self.yx_separable:
            Wy = Wxy_init * torch.randn((n_neurons, rank, Ly))
            Wx = Wxy_init * torch.randn((n_neurons, rank, Lx))       
            if y_init is not None: 
                Wy[np.arange(n_neurons), :, y_init] += Wxy_gabor_init
            if x_init is not None: 
                Wx[np.arange(n_neurons), :, x_init] += Wxy_gabor_init
            self.Wy = nn.Parameter(Wy)
            self.Wx = nn.Parameter(Wx)
        else:
            Wyx = .01 * torch.randn((n_neurons, Ly, Lx))
            Wyx[np.arange(n_neurons), y_init, x_init] += Wxy_gabor_init
            self.Wyx = nn.Parameter(Wyx)
            
        Wc = Wc_coef * torch.randn((n_neurons, rank, n_conv))
        if (c_init is not None) and (n_neurons > 1):
            Wc[np.arange(n_neurons), 0, c_init] += 0.5 * torch.from_numpy(coef_init.astype('float32'))
        self.Wc = nn.Parameter(Wc)
        if rank > 1:
            self.bias = nn.Parameter(torch.from_numpy(bias_init.astype('float32'))) if bias_init is not None else nn.Parameter(torch.zeros((rank, n_neurons)))
        else:
            self.bias = nn.Parameter(torch.from_numpy(bias_init.astype('float32'))) if bias_init is not None else nn.Parameter(torch.zeros(n_neurons))
        self.use_poisson = poisson
        self.rank = rank
    
    def forward(self, conv):
        if self.yx_separable:
            if self.rank > 1:
                pred = torch.einsum('nrc, nky, icyx, nkx->irn', self.Wc, self.Wy, conv, self.Wx)
            else:
                pred = torch.einsum('nrc, nry, icyx, nrx->in', self.Wc, self.Wy, conv, self.Wx)
            pred = pred.add(self.bias)
            pred = self.activation(pred)
            if self.rank > 1:
                pred = pred.sum(axis=1)
        else:
            pred = torch.einsum('nc, icyx, nyx->in', self.Wc, conv, self.Wyx)
        return pred
        
    def l1_norm(self):
        return self.Wc.abs().sum(axis=(1, 2))
    
    def l2_norm(self):
        Wc_l2 = (self.Wc**2).sum(axis=(1, 2))
        Wy_l2 = (self.Wy**2).sum(axis=(1, 2))
        Wx_l2 = (self.Wx**2).sum(axis=(1, 2))
        return Wc_l2 + Wy_l2 + Wx_l2
    
    def hoyer_square(self):
        wc_l1 = self.Wc.abs().sum(axis=(1, 2))
        wc_l2 = (self.Wc**2).sum(axis=(1, 2))
        return wc_l1**2 / wc_l2


# ============================================================
# 4. ViT Encoder (full model)
# ============================================================

class ViTEncoder(nn.Module):
    """
    Full encoding model: DINOv3 core + spatial readout.
    Drop-in replacement for your CNN Encoder class.
    """
    def __init__(self, core, readout, preprocessor=None, 
                 loss_fun='poisson', device=torch.device('cuda')):
        super().__init__()
        self.core = core
        self.readout = readout
        self.preprocessor = preprocessor  # HuggingFace image processor
        self.loss_fun = loss_fun
        self.bias = 1e-12
        self.device = device

    def forward(self, pixel_values, detach_core=False):
        """
        Args:
            pixel_values: (B, 3, H, W) preprocessed by DINOv3 image processor
            detach_core: if True, stop gradients from flowing to core
        """
        x = self.core(pixel_values)
        if detach_core:
            x = x.detach()
        x = self.readout(x)
        x += 1 + self.bias  # ensure positive for Poisson loss
        return x
        
    def loss_function(self, spks_batch, spks_pred, l1_readout=0, l2_readout=0, hs_reg=0.0):
        if self.loss_fun == 'poisson':
            loss = (spks_pred - spks_batch * torch.log(spks_pred)).sum(axis=0)
        else:
            loss = ((spks_pred - spks_batch)**2).sum(axis=0)
        loss += l1_readout * self.readout.l1_norm()
        loss += l2_readout * self.readout.l2_norm()
        loss += hs_reg * self.readout.hoyer_square()
        loss = loss.mean()
        return loss

    def preprocess_images(self, images, device=None):
        """
        Preprocess raw grayscale images for DINOv3.
        
        Args:
            images: numpy array (B, H, W) uint8 or float grayscale images
            device: torch device
        Returns:
            pixel_values: (B, 3, 224, 224) tensor ready for DINOv3
        """
        if device is None:
            device = self.device
        
        # Convert grayscale to RGB PIL-like format
        from PIL import Image
        pil_images = []
        for img in images:
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            # Grayscale -> RGB
            rgb = np.stack([img, img, img], axis=-1)  # (H, W, 3)
            pil_images.append(Image.fromarray(rgb))
        
        # Use HuggingFace image processor
        inputs = self.preprocessor(images=pil_images, return_tensors="pt")
        return inputs['pixel_values'].to(device)

    @torch.no_grad()
    def responses(self, images, core=False, batch_size=8, device=None):
        """Get model responses for a batch of images."""
        if device is None:
            device = self.device
        nimg = images.shape[0]
        n_batches = int(np.ceil(nimg / batch_size))
        self.eval()
        activations = None
        
        for k in range(n_batches):
            inds = np.arange(k * batch_size, min(nimg, (k + 1) * batch_size))
            pixel_values = self.preprocess_images(images[inds], device=device)
            
            if core:
                acts = self.core(pixel_values).cpu().numpy()
            else:
                acts = self.forward(pixel_values).cpu().numpy()
            acts = acts.reshape(acts.shape[0], -1)
            if activations is None:
                activations = np.zeros((nimg, *acts.shape[1:]), 'float32')
            activations[inds] = acts
        return activations


# ============================================================
# 5. Build Functions
# ============================================================

def build_dinov3_model(n_neurons, input_Ly=66, input_Lx=130, 
                       model_name='facebook/dinov3-vits16-pretrain-lvd1689m',
                       extract_layers=None,
                       use_channel_proj=True, proj_dim=64,
                       freeze_backbone=True,
                       poisson=True, Wc_coef=0.01,
                       device=torch.device('cuda')):
    """
    Build a DINOv3-based encoding model.
    
    Args:
        n_neurons: number of neurons to predict
        input_Ly, input_Lx: original image size (for reference only)
        model_name: HuggingFace model name
        extract_layers: which transformer blocks to extract features from
            None = last layer only
            e.g., [8, 9, 10, 11] for last 4 layers of vits16 (12 layers)
        use_channel_proj: project high-dim features down with 1x1 conv
        proj_dim: target channel dim after projection
        freeze_backbone: if True, only train readout + projection
        poisson: use Poisson loss
        Wc_coef: readout channel weight init scale
        device: cuda or cpu
    
    Returns:
        model: ViTEncoder (with preprocessor attached)
    """
    # Load image processor
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    
    # Determine input size from processor config
    # DINOv3 default is 224x224
    if hasattr(image_processor, 'size'):
        if isinstance(image_processor.size, dict):
            input_size = (image_processor.size.get('height', 224), 
                         image_processor.size.get('width', 224))
        else:
            input_size = (224, 224)
    else:
        input_size = (224, 224)
    
    print(f"Image processor input size: {input_size}")
    
    # Build core
    core = DINOv3Core(
        model_name=model_name,
        input_size=input_size,
        extract_layers=extract_layers,
        freeze=freeze_backbone,
        use_channel_proj=use_channel_proj,
        proj_dim=proj_dim
    )
    
    # Determine readout input shape
    in_shape = (core.out_channels, core.H_patches, core.W_patches)
    print(f"Readout input shape (C, H, W): {in_shape}")
    
    # Build readout
    readout = Readout(
        in_shape, n_neurons, rank=1,
        yx_separable=True, bias_init=None,
        poisson=poisson, Wc_coef=Wc_coef
    )
    
    # Build full model
    loss_fun = 'poisson' if poisson else 'mse'
    model = ViTEncoder(core, readout, preprocessor=image_processor, 
                       loss_fun=loss_fun, device=device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    return model


# ============================================================
# 6. Training Loop
# ============================================================

def train_model(model, images_train, spks_train, images_val, spks_val,
                n_epochs=100, batch_size=64, lr=1e-3,
                l1_readout=0.0, l2_readout=1e-4, hs_reg=0.0,
                weight_decay=1e-5,
                device=torch.device('cuda'),
                scheduler_patience=5,
                early_stop_patience=15,
                verbose=True):
    """
    Train the encoding model.
    
    Args:
        model: ViTEncoder (with attached preprocessor)
        images_train: (N_train, H, W) numpy array of images
        spks_train: (N_train, N_neurons) numpy array of neural responses
        images_val: (N_val, H, W) validation images
        spks_val: (N_val, N_neurons) validation responses
        
    Returns:
        train_losses, val_losses: lists of per-epoch losses
    """
    model = model.to(device)
    
    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Optimizing {len(trainable_params)} parameter groups, "
          f"{sum(p.numel() for p in trainable_params):,} params")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=scheduler_patience, factor=0.5
    )
    
    n_train = images_train.shape[0]
    n_val = images_val.shape[0]
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    for epoch in range(n_epochs):
        # ---- Training ----
        model.train()
        if model.core.freeze:
            model.core.backbone.eval()  # keep BN in eval mode
        
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = int(np.ceil(n_train / batch_size))
        
        for b in range(n_batches):
            inds = perm[b * batch_size: min(n_train, (b + 1) * batch_size)]
            
            # Preprocess images using HuggingFace processor
            pixel_values = model.preprocess_images(images_train[inds], device=device)
            spks = torch.from_numpy(spks_train[inds]).float().to(device)
            
            pred = model.forward(pixel_values)
            loss = model.loss_function(spks, pred, l1_readout, l2_readout, hs_reg)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / n_batches)
        
        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        n_val_batches = int(np.ceil(n_val / batch_size))
        
        with torch.no_grad():
            for b in range(n_val_batches):
                inds = np.arange(b * batch_size, min(n_val, (b + 1) * batch_size))
                pixel_values = model.preprocess_images(images_val[inds], device=device)
                spks = torch.from_numpy(spks_val[inds]).float().to(device)
                
                pred = model.forward(pixel_values)
                loss = model.loss_function(spks, pred, l1_readout, l2_readout, hs_reg)
                val_loss += loss.item()
        
        val_losses.append(val_loss / n_val_batches)
        scheduler.step(val_losses[-1])
        
        # Early stopping
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
            print(f"Epoch {epoch:03d} | Train: {train_losses[-1]:.4f} | "
                  f"Val: {val_losses[-1]:.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Patience: {patience_counter}/{early_stop_patience}")
        
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    
    return train_losses, val_losses


# ============================================================
# 7. Fine-tuning with differential learning rates
# ============================================================

def finetune_model(model, images_train, spks_train, images_val, spks_val,
                   n_blocks_to_unfreeze=2,
                   n_epochs=50, batch_size=32,
                   backbone_lr=1e-5, readout_lr=1e-3,
                   l1_readout=0.0, l2_readout=1e-4, hs_reg=0.0,
                   weight_decay=1e-5,
                   device=torch.device('cuda'),
                   verbose=True):
    """
    Fine-tune the last n transformer blocks + readout with different LRs.
    """
    model = model.to(device)
    
    # Unfreeze last n blocks
    model.core.unfreeze_last_n_blocks(n_blocks_to_unfreeze)
    
    # Separate parameter groups
    backbone_params = []
    proj_params = []
    readout_params = []
    
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
        {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': weight_decay},
        {'params': proj_params, 'lr': backbone_lr * 5, 'weight_decay': weight_decay},
        {'params': readout_params, 'lr': readout_lr, 'weight_decay': weight_decay * 0.1},
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-7)
    
    print(f"Fine-tuning parameter groups:")
    print(f"  Backbone: {sum(p.numel() for p in backbone_params):,} params @ lr={backbone_lr}")
    print(f"  Projection: {sum(p.numel() for p in proj_params):,} params @ lr={backbone_lr*5}")
    print(f"  Readout: {sum(p.numel() for p in readout_params):,} params @ lr={readout_lr}")
    
    # Training loop (similar to train_model but with custom optimizer)
    n_train = images_train.shape[0]
    n_val = images_val.shape[0]
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_state = None
    
    for epoch in range(n_epochs):
        model.train()
        # Don't set backbone to eval - we want to update BN during fine-tuning
        
        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = int(np.ceil(n_train / batch_size))
        
        for b in range(n_batches):
            inds = perm[b * batch_size: min(n_train, (b + 1) * batch_size)]
            pixel_values = model.preprocess_images(images_train[inds], device=device)
            spks = torch.from_numpy(spks_train[inds]).float().to(device)
            
            pred = model.forward(pixel_values)
            loss = model.loss_function(spks, pred, l1_readout, l2_readout, hs_reg)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / n_batches)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = int(np.ceil(n_val / batch_size))
        with torch.no_grad():
            for b in range(n_val_batches):
                inds = np.arange(b * batch_size, min(n_val, (b + 1) * batch_size))
                pixel_values = model.preprocess_images(images_val[inds], device=device)
                spks = torch.from_numpy(spks_val[inds]).float().to(device)
                pred = model.forward(pixel_values)
                loss = model.loss_function(spks, pred, l1_readout, l2_readout, hs_reg)
                val_loss += loss.item()
        
        val_losses.append(val_loss / n_val_batches)
        
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        if verbose and (epoch % 5 == 0 or epoch == n_epochs - 1):
            print(f"Epoch {epoch:03d} | Train: {train_losses[-1]:.4f} | Val: {val_losses[-1]:.4f}")
    
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)
    
    return train_losses, val_losses


# ============================================================
# 8. Evaluation
# ============================================================

def evaluate_model(model, images, spks, spks_var=None, 
                   batch_size=64, device=torch.device('cuda')):
    """
    Compute FEV and correlation coefficient per neuron.
    """
    model.eval()
    model = model.to(device)
    
    n = images.shape[0]
    n_batches = int(np.ceil(n / batch_size))
    preds = []
    
    with torch.no_grad():
        for b in range(n_batches):
            inds = np.arange(b * batch_size, min(n, (b + 1) * batch_size))
            pixel_values = model.preprocess_images(images[inds], device=device)
            pred = model.forward(pixel_values)
            preds.append(pred.cpu().numpy())
    
    preds = np.concatenate(preds, axis=0)
    
    # FEV
    residual_var = np.var(spks - preds, axis=0)
    total_var = np.var(spks, axis=0)
    if spks_var is not None:
        fev = 1 - residual_var / np.maximum(total_var - spks_var, 1e-10)
    else:
        fev = 1 - residual_var / np.maximum(total_var, 1e-10)
    
    # CC per neuron
    cc = np.array([
        np.corrcoef(spks[:, i], preds[:, i])[0, 1] 
        for i in range(spks.shape[1])
    ])
    
    return fev, cc, preds


def plot_comparison(cnn_cc, vit_cc, vit_ft_cc=None, labels=None):
    """Scatter plot comparing model performances."""
    import matplotlib.pyplot as plt
    
    n_plots = 2 if vit_ft_cc is not None else 1
    fig, axes = plt.subplots(1, n_plots + 1, figsize=(5 * (n_plots + 1), 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    
    # CNN vs ViT frozen
    ax = axes[0]
    ax.scatter(cnn_cc, vit_cc, alpha=0.3, s=8, c='steelblue')
    lim = max(np.nanmax(cnn_cc), np.nanmax(vit_cc)) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.3)
    ax.set_xlabel('CNN minimodel CC')
    ax.set_ylabel('DINOv3 frozen CC')
    ax.set_title(f'Frozen backbone\nMedian ΔCC = {np.nanmedian(vit_cc - cnn_cc):.3f}')
    ax.set_aspect('equal')
    
    if vit_ft_cc is not None:
        # CNN vs ViT fine-tuned
        ax = axes[1]
        ax.scatter(cnn_cc, vit_ft_cc, alpha=0.3, s=8, c='coral')
        ax.plot([0, lim], [0, lim], 'k--', alpha=0.3)
        ax.set_xlabel('CNN minimodel CC')
        ax.set_ylabel('DINOv3 fine-tuned CC')
        ax.set_title(f'Fine-tuned backbone\nMedian ΔCC = {np.nanmedian(vit_ft_cc - cnn_cc):.3f}')
        ax.set_aspect('equal')
    
    # Histogram of differences
    ax = axes[-1]
    ax.hist(vit_cc - cnn_cc, bins=50, alpha=0.6, color='steelblue', label='Frozen')
    if vit_ft_cc is not None:
        ax.hist(vit_ft_cc - cnn_cc, bins=50, alpha=0.6, color='coral', label='Fine-tuned')
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('ΔCC (DINOv3 - CNN)')
    ax.set_ylabel('# neurons')
    ax.legend()
    ax.set_title('Improvement distribution')
    
    plt.tight_layout()
    return fig


# ============================================================
# 9. Example Usage
# ============================================================
"""
# ======== STEP 1: Build model (frozen backbone, train readout only) ========
model = build_dinov3_model(
    n_neurons=5000,
    input_Ly=66, input_Lx=130,
    model_name='facebook/dinov3-vits16-pretrain-lvd1689m',  # 384-dim, 12 layers
    extract_layers=None,        # use last layer only (simplest)
    use_channel_proj=True,      # project 384 -> 64 channels
    proj_dim=64,
    freeze_backbone=True,
    poisson=True,
    device=torch.device('cuda')
)

# ======== STEP 2: Train readout only ========
train_losses, val_losses = train_model(
    model,
    images_train, spks_train,    # your data: (N, H, W) and (N, n_neurons)
    images_val, spks_val,
    n_epochs=100, batch_size=64,
    lr=1e-3, l2_readout=1e-4,
    device=torch.device('cuda')
)

# ======== STEP 3: Evaluate ========
fev_frozen, cc_frozen, preds_frozen = evaluate_model(
    model, images_test, spks_test, device=torch.device('cuda')
)
print(f"Frozen backbone - Median CC: {np.median(cc_frozen):.3f}")

# ======== STEP 4: Save frozen model ========
torch.save(model.state_dict(), 'dinov3_frozen_readout.pt')

# ======== STEP 5: Fine-tune last 2 transformer blocks ========
train_losses_ft, val_losses_ft = finetune_model(
    model,
    images_train, spks_train,
    images_val, spks_val,
    n_blocks_to_unfreeze=2,
    n_epochs=50, batch_size=32,
    backbone_lr=1e-5,
    readout_lr=1e-3,
    device=torch.device('cuda')
)

# ======== STEP 6: Evaluate fine-tuned ========
fev_ft, cc_ft, preds_ft = evaluate_model(
    model, images_test, spks_test, device=torch.device('cuda')
)
print(f"Fine-tuned - Median CC: {np.median(cc_ft):.3f}")

# ======== STEP 7: Compare with CNN minimodel ========
# Assuming you have cnn_cc from your existing model
fig = plot_comparison(cnn_cc, cc_frozen, cc_ft)
fig.savefig('dinov3_vs_cnn_comparison.png', dpi=150, bbox_inches='tight')

# ======== OPTIONAL: Try multi-layer feature extraction ========
model_multi = build_dinov3_model(
    n_neurons=5000,
    model_name='facebook/dinov3-vits16-pretrain-lvd1689m',
    extract_layers=[8, 9, 10, 11],  # last 4 layers -> 384*4 = 1536 channels
    use_channel_proj=True,
    proj_dim=64,                     # project 1536 -> 64
    freeze_backbone=True,
    device=torch.device('cuda')
)

# ======== OPTIONAL: Try DINOv3 ConvNext (natively spatial) ========
# core = DINOv3ConvNextCore(
#     model_name='facebook/dinov3-convnext-tiny-pretrain-lvd1689m',
#     freeze=True, use_channel_proj=True, proj_dim=64
# )
"""
