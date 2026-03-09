import argparse
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import data, metrics
from utils.vit_encoding_model import (
    build_vit_encoder,
    finetune_vit,
    make_model_name,
    preprocess_images,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune ViT encoding model from frozen checkpoint (cluster-friendly)."
    )
    parser.add_argument("--mouse_id", type=int, default=3, help="mouse id (0-5)")
    parser.add_argument("--data_path", type=str, default="../data", help="path to data directory")

    parser.add_argument(
        "--frozen_path",
        type=str,
        default="../notebooks/checkpoints/vit_frozen",
        help="directory with frozen checkpoints",
    )
    parser.add_argument(
        "--ft_path",
        type=str,
        default="../notebooks/checkpoints/vit_finetune",
        help="directory to save fine-tuned checkpoints",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/dinov3-vits16-pretrain-lvd1689m",
        help="HuggingFace ViT model name",
    )
    parser.add_argument("--hf_token", type=str, default="", help="HuggingFace API token")
    parser.add_argument(
        "--token_type",
        type=str,
        default="patch",
        choices=["patch", "cls"],
        help="kept for checkpoint naming parity with notebooks",
    )
    parser.add_argument(
        "--extract_layers",
        type=int,
        nargs="+",
        default=[2],
        help="ViT layers to extract features from",
    )

    parser.add_argument("--n_blocks_to_unfreeze", type=int, default=3)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--readout_lr", type=float, default=1e-3)
    parser.add_argument("--l2_readout", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1)

    return parser.parse_args()


def _resolve_path(path_str):
    if os.path.isabs(path_str):
        return path_str
    return os.path.abspath(os.path.join(SCRIPT_DIR, path_str))


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    data_path = _resolve_path(args.data_path)
    frozen_path = _resolve_path(args.frozen_path)
    ft_path = _resolve_path(args.ft_path)
    os.makedirs(ft_path, exist_ok=True)

    mouse_id = args.mouse_id

    # ===== Data =====
    img = data.load_images(
        data_path,
        mouse_id,
        file=data.img_file_name[mouse_id],
        crop=False,
        normalize=False,
    )
    print(
        "raw img shape:",
        img.shape,
        " dtype:",
        img.dtype,
        " range: [%.1f, %.1f]" % (img.min(), img.max()),
    )

    img = preprocess_images(img, batch_size=2000)
    print("preprocessed img shape:", img.shape)
    print("  mean=%.4f  std=%.4f  range=[%.2f, %.2f]" % (img.mean(), img.std(), img.min(), img.max()))

    fname = "%s_nat60k_%s.npz" % (data.db[mouse_id]["mname"], data.db[mouse_id]["datexp"])
    spks, istim_train, istim_test, xpos, ypos, spks_rep_all = data.load_neurons(
        file_path=os.path.join(data_path, fname),
        mouse_id=mouse_id,
    )
    n_stim, n_neurons = spks.shape
    print("spks shape:", spks.shape)

    itrain, ival = data.split_train_val(istim_train, train_frac=0.9)
    spks, spks_rep_all = data.normalize_spks(spks, spks_rep_all, itrain)

    ineur = np.arange(0, n_neurons)
    spks_train = torch.from_numpy(spks[itrain][:, ineur])
    spks_val = torch.from_numpy(spks[ival][:, ineur])

    img_train = torch.from_numpy(img[istim_train][itrain]).to(device)
    img_val = torch.from_numpy(img[istim_train][ival]).to(device)
    img_test = torch.from_numpy(img[istim_test]).to(device)

    print("img_train:", img_train.shape)
    print("spks_train:", spks_train.shape)

    # ===== Build model =====
    model = build_vit_encoder(
        n_neurons=len(ineur),
        model_name=args.model_name,
        vit_input_size=(64, 128),
        out_spatial_size=(4, 8),
        extract_layers=args.extract_layers,
        use_channel_proj=False,
        proj_dim=64,
        freeze_backbone=True,
        poisson=True,
        hf_token=args.hf_token,
        device=device,
    )

    # ===== Load frozen checkpoint =====
    frozen_filename = make_model_name(
        data.mouse_names[mouse_id],
        data.exp_date[mouse_id],
        args.model_name,
        args.token_type,
        args.extract_layers,
    )
    frozen_ckpt = os.path.join(frozen_path, frozen_filename)

    if os.path.exists(frozen_ckpt):
        model.load_state_dict(torch.load(frozen_ckpt, map_location=device))
        print("Loaded frozen checkpoint:", frozen_ckpt)
    else:
        raise FileNotFoundError(
            "Frozen checkpoint not found. Run frozen training first. Missing: " + frozen_ckpt
        )

    # ===== Fine-tune =====
    ft_filename = frozen_filename.replace(
        ".pt", f"_ft{args.n_blocks_to_unfreeze}blocks.pt"
    )
    ft_ckpt = os.path.join(ft_path, ft_filename)
    print("Fine-tune checkpoint:", ft_ckpt)

    if not os.path.exists(ft_ckpt):
        best_state = finetune_vit(
            model,
            spks_train,
            spks_val,
            img_train,
            img_val,
            n_blocks_to_unfreeze=args.n_blocks_to_unfreeze,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            backbone_lr=args.backbone_lr,
            readout_lr=args.readout_lr,
            l2_readout=args.l2_readout,
            clamp=True,
            patience=args.patience,
            device=device,
        )
        torch.save(best_state, ft_ckpt)
        print("Saved fine-tuned model:", ft_ckpt)

    model.load_state_dict(torch.load(ft_ckpt, map_location=device))
    print("Loaded fine-tuned model:", ft_ckpt)

    # ===== Evaluate fine-tuned =====
    model.eval()
    ft_pred = []
    with torch.no_grad():
        for k in range(0, img_test.shape[0], args.batch_size):
            pred = model(img_test[k : k + args.batch_size])
            ft_pred.append(pred.cpu().numpy())
    ft_pred = np.vstack(ft_pred)

    ft_fev, ft_feve = metrics.feve(spks_rep_all, ft_pred)

    threshold = 0.15
    valid = np.where(ft_fev > threshold)[0]
    ft_feve_mean = np.mean(ft_feve[ft_fev > threshold])
    print(f"Valid neurons (FEV > {threshold}): {len(valid)} / {len(ft_fev)}")
    print(
        f"FEVE (ViT fine-tuned, {args.n_blocks_to_unfreeze} blocks): {ft_feve_mean:.4f}"
    )

    # ===== Evaluate frozen for comparison =====
    frozen_model = build_vit_encoder(
        n_neurons=len(ineur),
        model_name=args.model_name,
        vit_input_size=(64, 128),
        out_spatial_size=(4, 8),
        extract_layers=args.extract_layers,
        use_channel_proj=False,
        proj_dim=64,
        freeze_backbone=True,
        poisson=True,
        hf_token=args.hf_token,
        device=device,
    )
    frozen_model.load_state_dict(torch.load(frozen_ckpt, map_location=device))
    frozen_model.eval()

    frozen_pred = []
    with torch.no_grad():
        for k in range(0, img_test.shape[0], args.batch_size):
            frozen_pred.append(frozen_model(img_test[k : k + args.batch_size]).cpu().numpy())
    frozen_pred = np.vstack(frozen_pred)

    frozen_fev, frozen_feve = metrics.feve(spks_rep_all, frozen_pred)
    frozen_feve_mean = np.mean(frozen_feve[frozen_fev > threshold])
    print(f"FEVE (ViT frozen): {frozen_feve_mean:.4f}")

    # ===== Save summary =====
    res_fname = os.path.join(ft_path, f"vit_finetune_{data.mouse_names[mouse_id]}_result.txt")
    with open(res_fname, "a") as f:
        f.write(f"ft_ckpt={ft_ckpt}\n")
        f.write(f"frozen_ckpt={frozen_ckpt}\n")
        f.write(f"extract_layers={args.extract_layers}\n")
        f.write(f"n_blocks_to_unfreeze={args.n_blocks_to_unfreeze}\n")
        f.write(f"FEVE_finetuned={ft_feve_mean*100:.4f}%\n")
        f.write(f"FEVE_frozen={frozen_feve_mean*100:.4f}%\n")
        f.write(f"Valid neurons: {len(valid)} / {len(ft_fev)}\n\n")
    print("Results saved to:", res_fname)


if __name__ == "__main__":
    main()
