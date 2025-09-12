import torch
from cs336_basics.blocks import TransformerLM
from cs336_basics.loss import AdamW
import numpy as np
from cs336_basics.data_loading import load_data
from cs336_basics.loss import cross_entropy_loss
from cs336_basics.checkpoint import save_checkpoint
from pathlib import Path
import os, csv
import random
from cs336_basics.gradient_clipping import clip_gradients_


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train(
    *,
    # model params
    d_model: int,
    num_heads: int,
    d_ff: int,
    vocab_size: int,
    num_layers: int,
    theta: float | None = None,  # RoPE
    # optimizer params
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    eps: float,
    # train config
    batch_size: int,
    context_length: int,
    num_batches: int,
    checkpoint_freq_in_batches: int,
    log_freq_in_batches: int,
    val_fraction: float,
    seed: int,
    gradient_clipping_max_l2_norm: float,
    # other
    device: str,
    dataset_filename: str,
    savedir: str,
) -> None:
    set_seed(seed)

    savedir = Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)

    model = TransformerLM(
        d_model,
        num_heads,
        d_ff,
        vocab_size,
        context_length,
        num_layers,
        theta,
        device,
        torch.float,
    )
    optimizer = AdamW(model.parameters(), lr, weight_decay, betas, eps)
    data = np.memmap(dataset_filename, dtype=np.int16, mode="r")
    val_size = int(len(data) * val_fraction)
    train_data, val_data = data[:-val_size], data[-val_size:]

    with open(savedir / "loss.csv", mode="w", buffering=1, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch", "train_loss", "val_loss"])
        for iteration in range(num_batches):
            model.train()
            optimizer.zero_grad()

            train_inputs, train_targets = load_data(
                train_data, batch_size, context_length, device
            )
            logits = model(train_inputs)
            train_loss = cross_entropy_loss(logits, train_targets)

            train_loss.backward()
            clip_gradients_(model.parameters(), gradient_clipping_max_l2_norm)
            optimizer.step()

            if iteration % checkpoint_freq_in_batches == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    iteration,
                    os.fspath(savedir / f"batch_{iteration}.pt"),
                )
            if iteration % log_freq_in_batches == 0:
                model.eval()
                with torch.no_grad():
                    val_inputs, val_targets = load_data(
                        val_data, batch_size, context_length, device
                    )
                    logits = model(val_inputs)
                    val_loss = cross_entropy_loss(logits, val_targets)

                writer.writerow([iteration, train_loss.item(), val_loss.item()])
                f.flush()

    save_checkpoint(
        model, optimizer, iteration, os.fspath(savedir / f"final_{iteration}.pt")
    )
