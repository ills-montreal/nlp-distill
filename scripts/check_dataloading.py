import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from utils.data_multifiles import (
    MultiTeacherAlignedEmbeddingDataset,
    make_aligned_collate_fn,
)

logging.basicConfig(level=logging.INFO)

# os make WANDB_MODE offline
os.environ["WANDB_MODE"] = "offline"
wandb.init(project="textdistill-4")


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="bert-base-uncased")

    # teachers names
    parser.add_argument(
        "--teachers",
        type=Path,
        nargs="+",
    )

    parser.add_argument("--embedding_dimension", type=int, default=1024)

    # epoch
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    # gradient accumulation
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--train_normalized", action="store_true", default=False)

    # lr
    parser.add_argument("--lr", type=float, default=1e-4)

    return parser.parse_args()


@dataclass(frozen=True)
class KernelArg:
    average: str = "var"
    cov_diagonal: str = "var"
    cov_off_diagonal: str = ""

    optimize_mu: bool = False
    cond_modes: int = 1
    use_tanh: bool = True
    init_std: float = 0.01
    ff_residual_connection: bool = False
    ff_activation: str = "relu"
    ff_layer_norm: bool = True
    ff_layers: int = 2
    ff_dim_hidden: int = 0


def main():

    args = parse_arguments()

    # Load the model

    model_path = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dataset = MultiTeacherAlignedEmbeddingDataset(teachers_path=args.teachers)

    teacher_dims = [ds.embeddings.shape[-1] for ds in dataset.datasets]
    collate_fn = make_aligned_collate_fn(tokenizer, teachers_dims=teacher_dims)

    logging.log(logging.INFO, f"Dataset: {dataset}")

    def worker_init_fn(worker_id):
        worker_infos = torch.utils.data.get_worker_info()
        worker_id, n_workers, dataset = (
            worker_infos.id,
            worker_infos.num_workers,
            worker_infos.dataset,
        )
        dataset.roll(worker_id=worker_id, n_workers=n_workers)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=6,
        timeout=10000,
        prefetch_factor=256,
    )

    for k, batch in enumerate(dataloader):
        if k >= 100:
            break
        print(batch)


if __name__ == "__main__":
    main()
