import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import lightning as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint, StochasticWeightAveraging
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import LightningEnvironment
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from utils.DistributedFriendlyKernel import GaussianCondKernel
from utils.data_multifiles2 import (
    MultiTeacherAlignedEmbeddingDataset,
    make_aligned_collate_fn,
    ExtractedSubSet,
)
from utils.pl_model_snowflakes import (
    DistilledEmbedderPLModelAlignedInputs,
    DistilledEmbedderPLmodelAlignedInputsMSE,
    DistilledEmbedderPLmodelAlignedInputsCosineSimilarity,
)

torch.multiprocessing.set_sharing_strategy("file_system")

logging.basicConfig(level=logging.INFO)

# os make WANDB_MODE offline
os.environ["WANDB_MODE"] = "offline"


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, default="bert-base-uncased")

    # teachers names
    parser.add_argument(
        "--teachers",
        type=Path,
        nargs="+",
    )

    parser.add_argument("--embedding_dimension", type=int, default=None)

    # epoch
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    # gradient accumulation
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--train_normalized", action="store_true", default=False)

    # lr
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--MSE", action="store_true", default=False)
    parser.add_argument("--cosine_similarity", action="store_true", default=False)

    parser.add_argument("--experiment_name", type=str, default="default")

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
    ff_layers: int = 3
    ff_dim_hidden: int = 0


def main():

    args = parse_arguments()

    # cosine and mse are exclusive
    assert not (
        args.MSE and args.cosine_similarity
    ), "MSE and cosine_similarity are exclusive"

    # Load the model

    model_path = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    # fake input
    if args.embedding_dimension is None:
        input_ids = torch.randint(0, 1000, (1, 12))
        attention_mask = torch.ones_like(input_ids)
        output = model(input_ids, attention_mask=attention_mask)
        args.embedding_dimension = output[0].shape[-1]

    tokenizer.model_max_length = 512
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    kernel_arg = KernelArg()

    dataset = MultiTeacherAlignedEmbeddingDataset(
        teachers_paths=args.teachers, max_samples=100000 if args.test else -1
    )
    eval_set = ExtractedSubSet(
        dataset,
        [i for i in range(100)],
        # list(np.random.randint(len(dataset), size=(100,))),
    )

    teacher_dims = dataset.teachers_embeddings_dims()
    collate_fn = make_aligned_collate_fn(tokenizer, teachers_dims=teacher_dims)

    logging.log(logging.INFO, f"Dataset: {dataset}")

    if args.MSE:
        # make simple feedforward model using
        teachers_kernels = [
            nn.Sequential(
                nn.Linear(args.embedding_dimension, args.embedding_dimension),
                nn.ReLU(),
                nn.Linear(args.embedding_dimension, args.embedding_dimension),
                nn.ReLU(),
                nn.Linear(args.embedding_dimension, d),
            )
            for d in dataset.teachers_embeddings_dims()
        ]

    else:
        teachers_kernels = [
            GaussianCondKernel(kernel_arg, zc_dim=args.embedding_dimension, zd_dim=d)
            for d in dataset.teachers_embeddings_dims()
        ]

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
        num_workers=4,
        timeout=10000,
        prefetch_factor=256,
        worker_init_fn=worker_init_fn,
    )

    eval_data_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        timeout=10000,
        prefetch_factor=256,
    )

    if args.MSE:
        plModel = DistilledEmbedderPLmodelAlignedInputsMSE(
            model=model,
            teachers_kernels=teachers_kernels,
            lr=args.lr,
            train_normalized=args.train_normalized,
        )
    elif args.cosine_similarity:
        plModel = DistilledEmbedderPLmodelAlignedInputsCosineSimilarity(
            model=model,
            teachers_kernels=teachers_kernels,
            lr=args.lr,
            train_normalized=args.train_normalized,
        )
    else:
        plModel = DistilledEmbedderPLModelAlignedInputs(
            model=model,
            teachers_kernels=teachers_kernels,
            lr=args.lr,
            train_normalized=args.train_normalized,
        )

    # logging.log(logging.INFO, f"PL Model: {plModel}")

    # dump starting model
    # torch.save(plModel.state_dict(), f"checkpoints/{wandb.run.id}/init_model.pth")
    # plModel = torch.compile(plModel)

    # wandb_logger = WandbLogger(log_model=False, project="textdistill-5", id="ptg6gk4k")

    def sanitize_model_name(name):
        return name.replace("/", "_")

    wandb_logger = WandbLogger(
        log_model=False,
        project=args.experiment_name,
        name=sanitize_model_name(args.model_name),
        id=args.experiment_name + "_" + sanitize_model_name(args.model_name),
        resume="allow",
    )

    # wandb_logger.experiment.id = "ptg6gk4k"

    # wandb_logger.experiment.config.update(vars(args))

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{args.experiment_name}_{sanitize_model_name(args.model_name)}-{'MSE' if args.MSE else 'NLL'}",
        monitor="train_nll" if not args.MSE else "train_mse",
        every_n_train_steps=2000,
        mode="min",
        save_top_k=-1,
        save_last=True,
        auto_insert_metric_name=True,
    )

    # stochastic_weight_averaging = StochasticWeightAveraging()

    logging.log(logging.INFO, f"Make the trainer")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        val_check_interval=1000,
        check_val_every_n_epoch=None,
        accelerator="cuda",
        devices=1,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        log_every_n_steps=1,
        enable_progress_bar=True,
        logger=wandb_logger,
        plugins=[LightningEnvironment()],
        precision="bf16-mixed",
        callbacks=[checkpoint_callback],  # , stochastic_weight_averaging],
        enable_checkpointing=True,
        # strategy=DDPStrategy(find_unused_parameters=False),
        # strategy=DeepSpeedStrategy(),
        # strategy="ddp",
        fast_dev_run=16 if args.test else False,
        gradient_clip_val=0.5,
    )

    logging.log(logging.INFO, f"Start training")
    trainer.fit(
        plModel,
        train_dataloaders=dataloader,
        val_dataloaders=eval_data_loader,
        ckpt_path="last",
    )


if __name__ == "__main__":
    main()
