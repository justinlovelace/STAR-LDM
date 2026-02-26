import math
import os
import random
from collections import OrderedDict
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from ema_pytorch import EMA
from omegaconf import OmegaConf, open_dict
from transformers import get_scheduler

from star_ldm.data.dataset_utils import (
    DataCollatorWithDiffusionTokens,
    get_c4_validation_dataset,
    get_fineweb_streaming_dataset,
)


def cycle(dl):
    while True:
        for data in dl:
            yield data


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def compute_grad_norm(parameters):
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return None
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), p=2) for p in parameters]), p=2
    ).item()
    return total_norm


def separate_weight_decayable_params(params):
    no_wd_params = [param for param in params if param.ndim < 2]
    wd_params = [param for param in params if param not in set(no_wd_params)]
    return wd_params, no_wd_params


def get_adamw_optimizer(params, lr, betas, weight_decay, eps=1e-8):
    params = list(params)
    wd_params, no_wd_params = separate_weight_decayable_params(params)
    param_groups = [
        {'params': wd_params},
        {'params': no_wd_params, 'weight_decay': 0},
    ]
    return AdamW(param_groups, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)


def _get_linear_wsd_scheduler_lambda(
    current_step: int,
    *,
    num_total_steps: int,
    num_warmup_steps: int,
    decay_ratio: float = 0.1,
    min_lr_ratio: float = 0,
):
    num_decay_steps = int(num_total_steps * decay_ratio)
    num_stable_steps = num_total_steps - num_warmup_steps - num_decay_steps

    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    if current_step < num_warmup_steps + num_stable_steps:
        return 1.0
    if current_step < num_total_steps:
        progress = float(current_step - num_warmup_steps - num_stable_steps) / float(
            max(1, num_decay_steps)
        )
        return max(min_lr_ratio, 1.0 - ((1.0 - min_lr_ratio) * progress))
    return min_lr_ratio


def get_linear_wsd_schedule(
    optimizer,
    num_total_steps: int,
    num_warmup_steps: int,
    decay_ratio: float = 0.1,
    min_lr_ratio: float = 0,
    last_epoch: int = -1,
):
    """Warmup-Stable-Decay learning rate schedule (https://arxiv.org/abs/2502.02737)."""
    lr_lambda = partial(
        _get_linear_wsd_scheduler_lambda,
        num_total_steps=num_total_steps,
        num_warmup_steps=num_warmup_steps,
        decay_ratio=decay_ratio,
        min_lr_ratio=min_lr_ratio,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def exists(x):
    return x is not None


class TransfusionTrainer:
    def __init__(
        self,
        args,
        model,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        lr_schedule='cosine',
        num_warmup_steps=500,
        clip_grad_norm=1.0,
        adam_betas=(0.9, 0.99),
        weight_decay=0.01,
        eval_every=50,
        save_every=5000,
        step_ckpt_every=50000,
        mixed_precision='bf16',
        diffusion_mse_lambda=5.0,
    ):
        super().__init__()
        self.args = args
        self.clip_grad_norm = clip_grad_norm
        self.diffusion_mse_lambda = diffusion_mse_lambda

        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            log_with='wandb',
        )
        self.num_devices = self.accelerator.num_processes
        self.accelerator.print(f'Using {self.num_devices} devices')

        with open_dict(args):
            args.num_devices = self.num_devices

        if args.get('output_dir', None) is None:
            with open_dict(args):
                args.output_dir = os.path.join(
                    args.get('save_dir', 'saved_models'),
                    args.dataset_name,
                    args.wandb_name,
                )
            if self.accelerator.is_main_process:
                os.makedirs(args.output_dir, exist_ok=True)

        if self.accelerator.is_main_process:
            with open(os.path.join(args.output_dir, 'args.yaml'), 'w') as f:
                OmegaConf.save(config=args, f=f)
            results_folder = args.output_dir
            run = 'star-ldm-training'
            self.accelerator.init_trackers(
                run,
                config=OmegaConf.to_container(args),
                init_kwargs={"wandb": {"dir": results_folder, "name": args.wandb_name}},
            )

        self.model = model
        if self.accelerator.is_main_process:
            self.ema = EMA(
                model, beta=0.999, update_every=10, power=3 / 4, update_after_step=1000
            )
            self.ema.to(self.accelerator.device)

        self.eval_every = eval_every
        self.save_every = save_every
        self.step_ckpt_every = step_ckpt_every

        self.train_batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        # Dataset and dataloader
        self.num_diffusion_tokens = args.prompt_generator.prompt_length
        lm_name = args.train.lm_name

        dataset = get_fineweb_streaming_dataset(
            tokenizer_name=lm_name,
            subset=args.get('fineweb_subset', 'sample-10BT'),
        )

        diffusion_data_collator = DataCollatorWithDiffusionTokens(
            self.model.tokenizer,
            num_diffusion_tokens=self.num_diffusion_tokens,
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            drop_last=True,
            collate_fn=diffusion_data_collator,
            num_workers=4,
            prefetch_factor=8,
        )

        val_dataset = get_c4_validation_dataset(tokenizer_name=lm_name)
        val_collator = DataCollatorWithDiffusionTokens(
            self.model.tokenizer,
            num_diffusion_tokens=self.num_diffusion_tokens,
            validation=True,
        )
        self.val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.train_batch_size,
            drop_last=True,
            collate_fn=val_collator,
        )
        self.val_iter = cycle(self.val_dataloader)

        # Optimizer
        self.opt = get_adamw_optimizer(
            self.model.parameters(),
            lr=train_lr,
            betas=adam_betas,
            weight_decay=weight_decay,
        )

        # LR scheduler
        if lr_schedule == 'cosine':
            self.lr_scheduler = get_scheduler(
                lr_schedule,
                optimizer=self.opt,
                num_warmup_steps=num_warmup_steps * self.num_devices,
                num_training_steps=train_num_steps * self.num_devices,
            )
        elif lr_schedule == 'wsd':
            self.lr_scheduler = get_linear_wsd_schedule(
                optimizer=self.opt,
                num_total_steps=train_num_steps * self.num_devices,
                num_warmup_steps=num_warmup_steps * self.num_devices,
                decay_ratio=0.1,
            )
        else:
            raise ValueError(f'Unknown lr_schedule: {lr_schedule}')

        # Output folder
        if self.accelerator.is_main_process:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        self.step = 0

        # Auto-resume from checkpoint
        ckpt_path = os.path.join(args.output_dir, 'model.pt')
        if os.path.exists(ckpt_path):
            self.accelerator.print(f"Resuming training from {ckpt_path}")
            self.load(file_path=args.output_dir)

        # Prepare with accelerator
        device_placement = [True, True, False, True]
        self.model, self.opt, self.dataloader, self.lr_scheduler = (
            self.accelerator.prepare(
                self.model, self.opt, self.dataloader, self.lr_scheduler,
                device_placement=device_placement,
            )
        )
        self.data_iter = cycle(self.dataloader)

        if self.accelerator.is_main_process:
            self.ema.to(self.accelerator.device)

    def save(self, step_ckpt=False):
        if not self.accelerator.is_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'scheduler': self.lr_scheduler.state_dict(),
        }
        torch.save(data, str(self.results_folder / 'model.pt'))
        if step_ckpt:
            torch.save(data, str(self.results_folder / f'model_{self.step}.pt'))

    def load(self, file_path=None):
        self.accelerator.print(f"Loading model from {file_path}")
        file_path = Path(file_path) if exists(file_path) else self.results_folder

        if file_path.suffix == '.pt':
            data = torch.load(str(file_path), map_location='cpu', weights_only=False)
        else:
            data = torch.load(str(file_path / 'model.pt'), map_location='cpu', weights_only=False)

        self.model.load_state_dict(data['model'], strict=False)
        self.opt.load_state_dict(data['opt'])

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data['ema'])

        self.step = data['step']
        self.lr_scheduler.load_state_dict(data['scheduler'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss_nll = 0.0
                total_loss_diffusion = 0.0
                total_unweighted_loss_diffusion = 0.0
                total_loss = 0.0

                for i in range(self.gradient_accumulate_every):
                    context = (
                        accelerator.no_sync(self.model)
                        if i < (self.gradient_accumulate_every - 1)
                        else nullcontext()
                    )

                    with context:
                        data = next(self.data_iter)
                        loss = self.model(
                            data['input_ids'].to(device),
                            labels=data['labels'].to(device),
                            continuation_text=data['continuation'],
                            diffusion_token_mask=data['diffusion_token_mask'].to(device),
                        )

                        nll_loss = loss['nll_loss']
                        diffusion_loss = loss['diffusion_loss']
                        unweighted_loss = loss['unweighted_diffusion_loss']
                        combined_loss = nll_loss + self.diffusion_mse_lambda * diffusion_loss
                        combined_loss = combined_loss / self.gradient_accumulate_every

                        total_loss_nll += nll_loss.item() / self.gradient_accumulate_every
                        total_loss_diffusion += diffusion_loss.item() / self.gradient_accumulate_every
                        total_unweighted_loss_diffusion += unweighted_loss.item() / self.gradient_accumulate_every
                        total_loss += combined_loss.item()

                        self.accelerator.backward(combined_loss)

                grad_norm = compute_grad_norm(self.model.parameters())
                if self.accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
                self.opt.step()
                self.lr_scheduler.step()
                self.opt.zero_grad(set_to_none=True)
                if self.accelerator.is_main_process:
                    self.ema.update()

                self.step += 1
                if accelerator.is_main_process and self.step % 10 == 0:
                    logs = {
                        "train/loss": total_loss,
                        "train/diffusion/loss": total_loss_diffusion,
                        "train/diffusion/unweighted_loss": total_unweighted_loss_diffusion,
                        "train/diffusion/weighted_loss_ema": self.accelerator.unwrap_model(self.model).get_weighted_loss().item(),
                        "train/learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "train/grad_norm": grad_norm,
                        "train/step": self.step,
                        "train/samples": self.step * self.train_batch_size * self.num_devices,
                        "train/nll/loss": total_loss_nll,
                        "train/nll/perplexity": math.exp(total_loss_nll),
                    }

                    if self.step % self.eval_every == 0:
                        self.model.eval()
                        val_data = next(self.val_iter)
                        with torch.no_grad():
                            val_loss = self.model(
                                val_data['input_ids'].to(device),
                                labels=val_data['labels'].to(device),
                                continuation_text=val_data['continuation'],
                                diffusion_token_mask=val_data['diffusion_token_mask'].to(device),
                            )

                            logs['val/loss'] = val_loss['nll_loss'].item() + self.diffusion_mse_lambda * val_loss['diffusion_loss'].item()
                            logs['val/diffusion/loss'] = val_loss['diffusion_loss'].item()
                            logs['val/diffusion/unweighted_loss'] = val_loss['unweighted_diffusion_loss'].item()
                            logs['val/nll/loss'] = val_loss['nll_loss'].item()
                            logs['val/nll/perplexity'] = math.exp(val_loss['nll_loss'].item())

                        self.model.train()
                        pbar.set_postfix(OrderedDict([
                            ('name', self.args.wandb_name),
                            ('weighted_loss', self.accelerator.unwrap_model(self.model).get_weighted_loss().item()),
                        ]))

                    if accelerator.is_main_process:
                        accelerator.log(logs, step=self.step)

                if self.step % self.save_every == 0:
                    self.save(step_ckpt=((self.step % self.step_ckpt_every) == 0))

                pbar.update(1)

            accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            self.save()
        accelerator.wait_for_everyone()
        accelerator.print('Training complete')
