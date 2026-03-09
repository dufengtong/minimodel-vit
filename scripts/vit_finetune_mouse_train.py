"""
Launch script: submits one bsub job per mouse for ViT fine-tuning.
Usage: python vit_finetune_mouse_train.py
"""
import os

mouse_names = ["L1_A5", "L1_A1", "FX9", "FX10", "FX8", "FX20"]


def main():
    # --- config ---
    mouse_ids = [3]
    model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    token_type = "patch"
    extract_layers_list = [[2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12]]

    n_blocks_to_unfreeze = 2
    max_epochs = 100
    patience = 5
    batch_size = 32
    backbone_lr = 1e-5
    readout_lr = 1e-3
    l2_readout = 1e-4
    seed = 1

    data_path = "../data"
    frozen_path = "../notebooks/checkpoints/vit_frozen"
    ft_path = "../notebooks/checkpoints/vit_finetune"

    hf_token = os.getenv("HF_TOKEN")

    for mouse_id in mouse_ids:
        output_save_path = f"outputs/vit_finetune/{mouse_names[mouse_id]}"
        os.makedirs(output_save_path, exist_ok=True)

        for extract_layers in extract_layers_list:
            extract_layers_str = " ".join(str(l) for l in extract_layers)
            layer_tag = "-".join(str(l) for l in extract_layers)

            prefix = (
                f"vit_finetune_{mouse_names[mouse_id]}_l{layer_tag}_"
                f"ft{n_blocks_to_unfreeze}b_seed{seed}"
            )

            bsub_cmd = (
                f'bsub -n 2 -q gpu_h100 -gpu "num=1" '
                f"-J {prefix} "
                f"-o {output_save_path}/{prefix}.out "
                f"-e {output_save_path}/{prefix}.err "
                f'"bash vit_finetune_mouse_script.sh '
                f"{mouse_id} "
                f'\\"{model_name}\\" '
                f"{token_type} "
                f'\\"{extract_layers_str}\\" '
                f"{n_blocks_to_unfreeze} "
                f"{max_epochs} "
                f"{patience} "
                f"{batch_size} "
                f"{backbone_lr} "
                f"{readout_lr} "
                f"{l2_readout} "
                f"{seed} "
                f'\\"{data_path}\\" '
                f'\\"{frozen_path}\\" '
                f'\\"{ft_path}\\" '
                f'\\"{hf_token}\\"'
                f'"'
            )
            print(bsub_cmd)
            os.system(bsub_cmd)


if __name__ == "__main__":
    main()
