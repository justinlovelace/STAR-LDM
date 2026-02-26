"""
Train STAR-LDM on FineWeb.

Usage:
    # Single GPU
    python scripts/train.py --config configs/train_fineweb.yaml

    # Multi-GPU with Accelerate
    accelerate launch scripts/train.py --config configs/train_fineweb.yaml

    # Override config values from CLI
    accelerate launch scripts/train.py --config configs/train_fineweb.yaml \
        train.learning_rate=1e-4 train.train_batch_size=8
"""

import argparse
import sys

from omegaconf import OmegaConf, open_dict

from star_ldm.models.transfusion import TransfusionGPT
from star_ldm.training.trainer import TransfusionTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train STAR-LDM')
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to YAML config file',
    )
    args, overrides = parser.parse_known_args()
    return args.config, overrides


def main():
    config_path, overrides = parse_args()

    # Load base config and apply CLI overrides
    args = OmegaConf.load(config_path)
    if overrides:
        cli_conf = OmegaConf.from_dotlist(overrides)
        args = OmegaConf.merge(args, cli_conf)

    # Build model
    model = TransfusionGPT(
        dataset_name=args.dataset_name,
        gpt2_model_name=args.train.lm_name,
        transfusion_cfg=args,
        scale_by_std=True,
        global_norm=args.train.global_norm,
    )

    with open_dict(args):
        args.trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Build trainer and run
    trainer = TransfusionTrainer(
        args=args,
        model=model,
        train_batch_size=args.train.train_batch_size,
        train_lr=args.train.learning_rate,
        train_num_steps=args.train.num_train_steps,
        lr_schedule=args.train.lr_schedule,
        num_warmup_steps=args.train.lr_warmup_steps,
        clip_grad_norm=args.train.clip_grad_norm,
        adam_betas=tuple(args.train.betas),
        weight_decay=args.train.weight_decay,
        gradient_accumulate_every=args.train.gradient_accumulation_steps,
        eval_every=args.train.eval_every,
        mixed_precision=args.train.mixed_precision,
        diffusion_mse_lambda=args.train.diffusion_mse_lambda,
    )

    trainer.train()


if __name__ == '__main__':
    main()
