import lightning as pl
import pandas as pd
import torch
from lightning.pytorch.utilities import grad_norm
from torch import nn
from torch.optim import AdamW


class DistilledEmbedderPLModelAlignedInputs(pl.LightningModule):
    def __init__(self, model, teachers_kernels, lr=1e-3, train_normalized=True):
        super().__init__()
        self.model = model

        self.teachers_kernels = nn.ModuleList(teachers_kernels)
        self.lr = lr
        self.train_normalized = train_normalized

        self.save_hyperparameters("lr")
        self.save_hyperparameters("train_normalized")

        self.validation_step_output = []

    def forward(self, x):
        return self.model(**x)

    def training_step(self, batch, batch_idx):
        # List[Tensor], hf model inputs dict, Tensor : (n_teachers, batch_size)
        targets, inputs, mask, text = batch

        output = self.model(**inputs)
        output = output.last_hidden_state[:, -1]

        negative_log_likelihood = []
        for k, teacher_kernel in enumerate(self.teachers_kernels):
            selected = torch.where(mask[k])
            output_k = output[selected]
            target_k = targets[k][selected]

            if output_k.shape[0] == 0:
                continue
            nll = -teacher_kernel.logpdf(output_k, target_k).mean()
            negative_log_likelihood.append((nll, targets[k].shape[-1]))

            self.log(
                f"train_nll_{k}",
                nll,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size=output_k.shape[0],
            )

        negative_log_likelihood_normalized = sum(
            nll / dim for nll, dim in negative_log_likelihood
        )
        negative_log_likelihood = sum(nll for nll, _ in negative_log_likelihood)

        self.log(
            "train_nll",
            negative_log_likelihood,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=sum(mask[k].sum() for k in range(len(mask))),
        )

        self.log(
            "train_nll_normalized",
            negative_log_likelihood_normalized,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=sum(mask[k].sum() for k in range(len(mask))),
        )

        # check pairwise distances inside the batch
        output = output.detach()
        pairwise_distances = torch.cdist(output, output, p=2).mean()
        self.log("train_pairwise_distances", pairwise_distances)

        pairwise_distances = torch.cdist(output, output, p=float("+inf")).mean()
        self.log(
            "train_pairwise_distances_linf",
            pairwise_distances,
            on_step=False,
            on_epoch=True,
        )

        if self.train_normalized:
            return negative_log_likelihood_normalized
        else:
            return negative_log_likelihood

    def validation_step(self, batch, batch_idx):
        # List[Tensor], hf model inputs dict, Tensor : (n_teachers, batch_size)
        targets, inputs, mask, text = batch

        output = self.model(**inputs)
        output = output.last_hidden_state[:, -1]

        negative_log_likelihood = []
        for k, teacher_kernel in enumerate(self.teachers_kernels):
            selected = torch.where(mask[k])
            output_k = output[selected]
            target_k = targets[k][selected]
            if output_k.shape[0] == 0:
                continue
            nll = -teacher_kernel.logpdf(output_k, target_k).mean()
            negative_log_likelihood.append((nll, targets[k].shape[-1]))

            self.log(
                f"val_nll_{k}",
                nll,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=output_k.shape[0],
            )

        negative_log_likelihood_normalized = sum(
            _nll / dim for _nll, dim in negative_log_likelihood
        )
        negative_log_likelihood = sum(nll for nll, _ in negative_log_likelihood)

        self.log(
            "val_nll",
            negative_log_likelihood,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=sum(mask[k].sum() for k in range(len(mask))),
        )

        self.log(
            "val_nll_normalized",
            negative_log_likelihood_normalized,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=sum(mask[k].sum() for k in range(len(mask))),
        )

        # check pairwise distances inside the batch
        output = output.detach()
        pairwise_distances = torch.cdist(output, output, p=2).mean()
        self.log(
            "val_pairwise_distances", pairwise_distances, on_step=False, on_epoch=True
        )

        pairwise_distances = torch.cdist(output, output, p=float("+inf")).mean()
        self.log(
            "val_pairwise_distances_linf",
            pairwise_distances,
            on_step=False,
            on_epoch=True,
        )

        self.validation_step_output.append((output.cpu(), text))

    def on_validation_epoch_end(self):
        texts, outputs = [x[1] for x in self.validation_step_output], [
            x[0] for x in self.validation_step_output
        ]
        outputs = torch.cat(outputs, dim=0).cpu()  # shape (n_samples, embedding_dim)

        # flatten list of texts
        texts = [text for batch in texts for text in batch]

        # make a dataframe with the dims as columns
        df = pd.DataFrame(
            outputs.numpy(), columns=[f"dim_{i}" for i in range(outputs.shape[1])]
        )
        df["text"] = texts
        # self.log({"val_embeddings": df})
        self.logger.log_table("val_embeddings", dataframe=df)

        self.validation_step_output = []

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)

        return optimizer


class DistilledEmbedderPLmodelAlignedInputsMSE(DistilledEmbedderPLModelAlignedInputs):
    def __init__(self, model, teachers_kernels, lr=1e-3, train_normalized=True):
        super().__init__(model, teachers_kernels, lr, train_normalized)

    def training_step(self, batch, batch_idx):
        # List[Tensor], hf model inputs dict, Tensor : (n_teachers, batch_size)
        targets, inputs, mask, text = batch

        output = self.model(**inputs)
        output = output.last_hidden_state[:, -1]

        mse = []
        for k, teacher_kernel in enumerate(self.teachers_kernels):
            selected = torch.where(mask[k])
            output_k = output[selected]
            target_k = targets[k][selected]

            if output_k.shape[0] == 0:
                continue

            reconstructed = teacher_kernel(output_k)
            mse_k = nn.functional.mse_loss(reconstructed, target_k)
            mse.append((mse_k, targets[k].shape[-1]))

            self.log(
                f"train_mse_{k}",
                mse_k,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size=output_k.shape[0],
            )

        mse_normalized = sum(mse_k / dim for mse_k, dim in mse)
        mse = sum(mse_k for mse_k, _ in mse)

        self.log(
            "train_mse",
            mse,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=sum(mask[k].sum() for k in range(len(mask))),
        )

        self.log(
            "train_mse_normalized",
            mse_normalized,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=sum(mask[k].sum() for k in range(len(mask))),
        )

        # check pairwise distances inside the batch
        output = output.detach()
        pairwise_distances = torch.cdist(output, output, p=2).mean()
        self.log("train_pairwise_distances", pairwise_distances)

        pairwise_distances = torch.cdist(output, output, p=float("+inf")).mean()
        self.log(
            "train_pairwise_distances_linf",
            pairwise_distances,
            on_step=False,
            on_epoch=True,
        )

        if self.train_normalized:
            return mse_normalized
        else:
            return mse

    def validation_step(self, batch, batch_idx):
        # List[Tensor], hf model inputs dict, Tensor : (n_teachers, batch_size)
        targets, inputs, mask, text = batch

        output = self.model(**inputs)
        output = output.last_hidden_state[:, -1]

        mse = []
        for k, teacher_kernel in enumerate(self.teachers_kernels):
            selected = torch.where(mask[k])
            output_k = output[selected]
            target_k = targets[k][selected]
            if output_k.shape[0] == 0:
                continue

            reconstructed = teacher_kernel(output_k)
            mse_k = nn.functional.mse_loss(reconstructed, target_k)

            self.log(
                f"val_mse_{k}",
                mse_k,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=output_k.shape[0],
            )

        mse_normalized = sum(mse_k / dim for mse_k, dim in mse)
        mse = sum(mse_k for mse_k, _ in mse)

        self.log(
            "val_mse",
            mse,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=sum(mask[k].sum() for k in range(len(mask))),
        )

        self.log(
            "val_mse_normalized",
            mse_normalized,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=sum(mask[k].sum() for k in range(len(mask))),
        )

        # check pairwise distances inside the batch
        output = output.detach()
        pairwise_distances = torch.cdist(output, output, p=2).mean()
        self.log(
            "val_pairwise_distances", pairwise_distances, on_step=False, on_epoch=True
        )

        pairwise_distances = torch.cdist(output, output, p=float("+inf")).mean()
        self.log(
            "val_pairwise_distances_linf",
            pairwise_distances,
            on_step=False,
            on_epoch=True,
        )

        self.validation_step_output.append((output.cpu(), text))

    def on_validation_epoch_end(self):
        texts, outputs = [x[1] for x in self.validation_step_output], [
            x[0] for x in self.validation_step_output
        ]
        outputs = torch.cat(outputs, dim=0).cpu()

        texts = [text for batch in texts for text in batch]

        df = pd.DataFrame(
            outputs.numpy(), columns=[f"dim_{i}" for i in range(outputs.shape[1])]
        )
        df["text"] = texts
        self.logger.log_table("val_embeddings", dataframe=df)

        self.validation_step_output = []
