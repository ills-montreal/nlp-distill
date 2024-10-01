from sys import argv

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling
from summarization.summarization_evaluation.cache_classifiers import tokenizer
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

from utils.pl_model_snowflakes import (
    AccessTeacherEstimationWrapper,
    DistilledEmbedderPLModelAlignedInputs,
)

import uniqid


def rmdir(directory):
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()


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


if __name__ == "__main__":

    ckpt_path = argv[1]
    output_dir = Path(argv[2])
    backbone_name = argv[3]
    teacher_dim = int(argv[4])
    loss = argv[5]

    if loss == "MSE":
        raise NotImplementedError("MSE loss not implemented yet")
    else:
        model = DistilledEmbedderPLModelAlignedInputs.load_from_checkpoint(ckpt_path)

    tokenizer = AutoTokenizer.from_pretrained(backbone_name)

    example = "Hey how are you"
    inputs = tokenizer(example, return_tensors="pt")
    outputs = model(**inputs)
    print(outputs.last_hidden_state[:, 0])

    pl_state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))

    # select only keys that starts with "model."
    state_dict = pl_state_dict["state_dict"]
    state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}

    model.load_state_dict(state_dict, strict=False)

    example = "Hey how are you"
    inputs = tokenizer(example, return_tensors="pt")
    outputs = model(**inputs)
    print(outputs.last_hidden_state[:, 0])
    print(outputs.last_hidden_state[:, 0].shape)
    embedding_dim = outputs.last_hidden_state[:, 0].shape[-1]

    # random temp directory
    temp_dir_path = Path("tmp") / uniqid.uniqid()

    # if exists, remove it
    if temp_dir_path.exists():
        rmdir(temp_dir_path)

    temp_dir_path.mkdir(exist_ok=True, parents=True)

    model.save_pretrained(str(temp_dir_path))
    tokenizer.save_pretrained(str(temp_dir_path))

    model_name = Path(ckpt_path).stem

    hf_backbone = SentenceTransformer(
        str(temp_dir_path), trust_remote_code=True, device="cpu"
    )

    print(hf_backbone)
    print(hf_backbone.get_sentence_embedding_dimension())

    print(hf_backbone.encode(example))
    print(hf_backbone.encode(example).shape)

    hf_backbone.save_pretrained(str(output_dir / model_name))
