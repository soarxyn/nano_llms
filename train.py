import torch
from torch.utils.data import DataLoader
from nano_llms.transformer import Transformer
from nano_llms.training import save_checkpoint, TinyStoriesDataset
from nano_llms.adamw import AdamW, lr_cosine_schedule, clip_grad
from nano_llms.ops import cross_entropy
from tqdm import trange

from dataclasses import dataclass

from jsonargparse import ArgumentParser


@dataclass
class DatasetParams:
    train_path: str
    valid_path: str
    batch_size: int


@dataclass
class TrainerParams:
    max_grad_norm: float
    max_steps: int
    validate_every_n_steps: int
    validation_steps: int


@dataclass
class SchedulerParams:
    max_lr: float
    min_lr: float
    warmup_steps: int


@torch.inference_mode()
def evaluate(model, valid_loader, validation_steps, device):
    model.eval()

    total_loss = 0.0

    for _ in trange(validation_steps):
        batch = next(iter(valid_loader))

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
    optimizer = AdamW(model.parameters())

    model.train()

    for idx in trange(cfg.trainer.max_steps):
        batch = next(iter(train_dataloader))

        current_lr = lr_cosine_schedule(
            idx,
            cfg.scheduler.max_lr,
            cfg.scheduler.min_lr,
            cfg.scheduler.warmup_steps,
            cfg.trainer.max_steps,
        )

        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        tokens, next_tokens = batch
        tokens = tokens.to(device, non_blocking=True)
        next_tokens = next_tokens.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(tokens)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), next_tokens.view(-1))

        loss.backward()
        clip_grad(model.parameters(), cfg.trainer.max_grad_norm)

        print(loss.item())

        optimizer.step()

        if idx % cfg.trainer.validate_every_n_steps == 0:
            valid_loss = evaluate(
                model, valid_dataloader, cfg.trainer.validation_steps, device
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", action="config")
    parser.add_class_arguments(Transformer, "transformer")
    parser.add_class_arguments(DatasetParams, "dataset")
    parser.add_class_arguments(TrainerParams, "trainer")
    parser.add_class_arguments(SchedulerParams, "scheduler")

    cfg = parser.parse_args()
    train(cfg)
