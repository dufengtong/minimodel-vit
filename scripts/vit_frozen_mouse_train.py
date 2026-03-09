"""
Launch script: submits one bsub job per mouse for frozen ViT training.
Usage: python vit_frozen_mouse_train.py
"""
import os
import numpy as np

mouse_names = ['L1_A5', 'L1_A1', 'FX9', 'FX10', 'FX8', 'FX20']

def main():
    # --- config ---
    mouse_ids      = [3]   # which mice to train
    model_name     = 'facebook/dinov3-vits16-pretrain-lvd1689m'
    token_type     = 'patch'
    extract_layers = [0]
    max_epochs     = 200
    patience       = 5
    batch_size     = 64
    lr             = 1e-3
    l2_readout     = 0.1
    seed           = 1
    data_path      = '../data'
    weight_path    = '../notebooks/checkpoints/vit_frozen'
    hf_token       = os.getenv("HF_TOKEN")                                 # <-- fill in your HuggingFace token

    

    for mouse_id in mouse_ids:
        output_save_path = f'outputs/vit_frozen/{mouse_names[mouse_id]}'
        os.makedirs(output_save_path, exist_ok=True)

        prefix = f'vit_frozen_{mouse_names[mouse_id]}_seed{seed}'

        for extract_layer in extract_layers:
            extract_layer = [extract_layer]  # ensure it's a list for the command-line argument
            extract_layers_str = ' '.join(str(l) for l in extract_layer)
            bsub_cmd = (
                f'bsub -n 4 -q gpu_h100 -gpu "num=1" '
                f'-J {prefix} '
                f'-o {output_save_path}/{prefix}.out '
                f'-e {output_save_path}/{prefix}.err '
                f'"bash vit_frozen_mouse_script.sh '
                f'{mouse_id} '
                f'\\"{model_name}\\" '
                f'{token_type} '
                f'\\"{extract_layers_str}\\" '
                f'{max_epochs} '
                f'{patience} '
                f'{batch_size} '
                f'{lr} '
                f'{l2_readout} '
                f'{seed} '
                f'\\"{data_path}\\" '
                f'\\"{weight_path}\\" '
                f'\\"{hf_token}\\"'
                f'"'
            )
            print(bsub_cmd)
            os.system(bsub_cmd)


if __name__ == '__main__':
    main()
