import math
from dataclasses import dataclass

import torch
import wandb
from jsonargparse import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import trange

from nano_llms.adamw import AdamW, clip_grad, lr_cosine_schedule
from nano_llms.ops import cross_entropy
from nano_llms.training import TinyStoriesDataset, save_checkpoint
from nano_llms.transformer import Transformer


@dataclass
class DatasetParams:
    train_path: str
    valid_path: str
    batch_size: int


@dataclass
class TrainerParams:
    max_grad_norm: float
    max_tokens: int
    validate_every_n_steps: int
    validation_steps: int
    checkpoint_every_n_steps: int
    checkpoint_path: str


@dataclass
class SchedulerParams:
    max_lr: float
    min_lr: float
    warmup_steps_pct: float
    base_batch_size: int


@dataclass
class OptimizerParams:
    betas: tuple[float, float]
    eps: float
    weight_decay: float


@torch.inference_mode()
def evaluate(model, valid_iter, validation_steps, device):
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        model.eval()

        total_loss = 0.0

        for _ in trange(validation_steps):
            batch = next(valid_iter)

            tokens, next_tokens = batch
            tokens = tokens.to(device, non_blocking=True)
            next_tokens = next_tokens.to(device, non_blocking=True)

            logits = model(tokens)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), next_tokens.view(-1))
            total_loss += loss.item()

        model.train()
        return total_loss / validation_steps


def train(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

    with wandb.init(project="nano-llms", config=cfg.as_dict()) as run:
        train_dataset = TinyStoriesDataset(
            cfg.dataset.train_path,
            cfg.dataset.batch_size,
            cfg.transformer.context_length,
            "cpu",
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=None,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=2,
        )

        valid_dataset = TinyStoriesDataset(
            cfg.dataset.valid_path,
            cfg.dataset.batch_size,
            cfg.transformer.context_length,
            "cpu",
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=None,
            pin_memory=True,
            num_workers=4,
            prefetch_factor=2,
        )

        model: Transformer = Transformer(**cfg.transformer.as_dict()).to(device)
        torch.compile(model)

        optimizer = AdamW(
            model.parameters(),
            betas=tuple(cfg.optimizer.betas),
            eps=cfg.optimizer.eps,
            weight_decay=cfg.optimizer.weight_decay,
        )

        run.watch(model, log="all")

        model.train()

        param_count = (
            sum([p.numel() for p in model.parameters() if p.requires_grad]) // 1e6
        )
        model_size = (
            sum(
                [
                    p.nelement() * p.element_size()
                    for p in model.parameters()
                    if p.requires_grad
                ]
            )
            / 1024**2
        )

        print(f"Total parameters: {param_count}M, model size: {model_size}MB")

        train_iter = iter(train_dataloader)
        valid_iter = iter(valid_dataloader)

        tokens_per_step: int = cfg.dataset.batch_size * cfg.transformer.context_length
        max_steps: int = cfg.trainer.max_tokens // tokens_per_step
        warmup_steps: int = int(max_steps * cfg.scheduler.warmup_steps_pct)
        effective_lr: float = cfg.scheduler.max_lr * math.sqrt(
            cfg.dataset.batch_size / cfg.scheduler.base_batch_size
        )

        run.config.update(
            {
                "max_steps": max_steps,
                "warmup_steps": warmup_steps,
                "effective_lr": effective_lr,
            }
        )

        for idx in trange(max_steps):
            batch = next(train_iter)

            current_lr = lr_cosine_schedule(
                idx,
                effective_lr,
                cfg.scheduler.min_lr,
                warmup_steps,
                max_steps,
            )

            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            tokens, next_tokens = batch
            tokens = tokens.to(device, non_blocking=True)
            next_tokens = next_tokens.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits = model(tokens)
                loss = cross_entropy(
                    logits.view(-1, logits.size(-1)), next_tokens.view(-1)
                )

            loss.backward()
            clip_grad(model.parameters(), cfg.trainer.max_grad_norm)

            optimizer.step()

            run.log(
                {
                    "train/loss": loss.item(),
                    "train/lr": current_lr,
                    "train_perplexity": math.exp(loss.item()),
                },
                step=idx,
            )

            if idx != 0:
                if idx % cfg.trainer.checkpoint_every_n_steps == 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        idx,
                        cfg.trainer.checkpoint_path + f"checkpoint-step={idx}.ckpt",
                    )

                if idx % cfg.trainer.validate_every_n_steps == 0:
                    valid_loss = evaluate(
                        model, valid_iter, cfg.trainer.validation_steps, device
                    )

                    run.log(
                        {
                            "val/loss": valid_loss,
                            "val/perplexity": math.exp(valid_loss),
                        },
                        step=idx,
                    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(Transformer, "transformer")
    parser.add_class_arguments(DatasetParams, "dataset")
    parser.add_class_arguments(TrainerParams, "trainer")
    parser.add_class_arguments(SchedulerParams, "scheduler")
    parser.add_class_arguments(OptimizerParams, "optimizer")

    cfg = parser.parse_args()
    train(cfg)
