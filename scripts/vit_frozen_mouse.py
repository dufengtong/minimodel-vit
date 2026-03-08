import os
import sys
import torch
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mouse_id',       type=int,   default=3,    help='mouse id (0-5)')
parser.add_argument('--data_path',      type=str,   default='../notebooks/data', help='path to data directory')
parser.add_argument('--weight_path',    type=str,   default='../notebooks_vit/checkpoints/vit_frozen', help='path to save model checkpoints')
parser.add_argument('--model_name',     type=str,   default='facebook/dinov3-vits16-pretrain-lvd1689m', help='HuggingFace ViT model name')
parser.add_argument('--hf_token',       type=str,   default='',   help='HuggingFace API token')
parser.add_argument('--token_type',     type=str,   default='patch', choices=['patch', 'cls'], help='token type: patch (spatial) or cls (global)')
parser.add_argument('--extract_layers', type=int,   default=[4],  nargs='+', help='ViT layers to extract features from (e.g. --extract_layers 4)')
parser.add_argument('--max_epochs',     type=int,   default=200,  help='max training epochs')
parser.add_argument('--patience',       type=int,   default=5,    help='early stopping patience')
parser.add_argument('--batch_size',     type=int,   default=64,   help='training batch size')
parser.add_argument('--lr',             type=float, default=1e-3, help='learning rate')
parser.add_argument('--l2_readout',     type=float, default=0.1,  help='L2 regularization on readout')
parser.add_argument('--seed',           type=int,   default=1,    help='random seed')
args = parser.parse_args()

np.random.seed(args.seed)

# ===== Data =====
from utils import data

mouse_id  = args.mouse_id
data_path = args.data_path

# Load images (no normalization; preprocess_images handles it)
img = data.load_images(data_path, mouse_id, file=data.img_file_name[mouse_id],
                       crop=False, normalize=False)
print('raw img shape:', img.shape, ' dtype:', img.dtype,
      ' range: [%.1f, %.1f]' % (img.min(), img.max()))

# Preprocess: 66x264 -> (N, 3, 64, 128), ImageNet-normalized
from utils.vit_encoding_model import preprocess_images
img = preprocess_images(img, batch_size=2000)
print('preprocessed img shape:', img.shape)

# Load neurons
fname = '%s_nat60k_%s.npz' % (data.db[mouse_id]['mname'], data.db[mouse_id]['datexp'])
spks, istim_train, istim_test, xpos, ypos, spks_rep_all = data.load_neurons(
    file_path=os.path.join(data_path, fname), mouse_id=mouse_id
)
n_stim, n_neurons = spks.shape
print('spks shape:', spks.shape)

# Split train / validation
itrain, ival = data.split_train_val(istim_train, train_frac=0.9)
print('itrain:', itrain.shape, ' ival:', ival.shape)

# Normalize neural data
spks, spks_rep_all = data.normalize_spks(spks, spks_rep_all, itrain)

# Prepare tensors
ineur       = np.arange(0, n_neurons)
spks_train  = torch.from_numpy(spks[itrain][:, ineur])
spks_val    = torch.from_numpy(spks[ival][:, ineur])
img_train   = torch.from_numpy(img[istim_train][itrain]).to(device)
img_val     = torch.from_numpy(img[istim_train][ival]).to(device)
img_test    = torch.from_numpy(img[istim_test]).to(device)

print('spks_train:', spks_train.shape)
print('spks_val:  ', spks_val.shape)
print('img_train: ', img_train.shape)
print('img_test:  ', img_test.shape)

# ===== Build ViT model (frozen backbone) =====
from utils.vit_encoding_model import build_vit_encoder, make_model_name

model = build_vit_encoder(
    n_neurons        = len(ineur),
    model_name       = args.model_name,
    vit_input_size   = (64, 128),
    out_spatial_size = (4, 8),
    extract_layers   = args.extract_layers,
    use_channel_proj = False,
    proj_dim         = 64,
    freeze_backbone  = True,
    poisson          = True,
    hf_token         = args.hf_token,
    device           = device,
)

# ===== Train readout =====
from utils.vit_encoding_model import train_readout

weight_path = args.weight_path
os.makedirs(weight_path, exist_ok=True)

model_filename = make_model_name(
    data.mouse_names[mouse_id], data.exp_date[mouse_id],
    args.model_name, args.token_type, args.extract_layers,
)
model_path = os.path.join(weight_path, model_filename)
print('Checkpoint:', model_path)

if not os.path.exists(model_path):
    best_state = train_readout(
        model,
        spks_train, spks_val,
        img_train,  img_val,
        max_epochs = args.max_epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        l2_readout = args.l2_readout,
        clamp      = True,
        patience   = args.patience,
        device     = device,
    )
    torch.save(best_state, model_path)
    print('Saved:', model_path)
else:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print('Loaded:', model_path)

# ===== Evaluate: FEVE on test set =====
from utils import metrics

model.eval()
n_test   = img_test.shape[0]
vit_pred = []
with torch.no_grad():
    for k in range(0, n_test, args.batch_size):
        pred = model(img_test[k:k + args.batch_size])
        vit_pred.append(pred.cpu().numpy())
vit_pred = np.vstack(vit_pred)
print('vit_pred:', vit_pred.shape, vit_pred.min(), vit_pred.max())

vit_fev, vit_feve = metrics.feve(spks_rep_all, vit_pred)

threshold = 0.15
valid     = np.where(vit_fev > threshold)[0]
print(f'Valid neurons (FEV > {threshold}): {len(valid)} / {len(vit_fev)}')
print(f'FEVE (test, ViT frozen): {np.mean(vit_feve[vit_fev > threshold]):.4f}')

# Save results
res_fname = os.path.join(weight_path, f'vit_frozen_{data.mouse_names[mouse_id]}_result.txt')
with open(res_fname, 'a') as f:
    f.write(f'{model_path}\n')
    f.write(f'FEVE(test)={vit_feve[vit_fev > threshold].mean()*100:.4f}%\n')
    f.write(f'Valid neurons: {len(valid)} / {len(vit_fev)}\n')
print('Results saved to:', res_fname)
