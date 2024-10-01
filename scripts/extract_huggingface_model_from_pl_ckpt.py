from sys import argv

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

import uniqid


def rmdir(directory):
    directory = Path(directory)
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()


if __name__ == "__main__":

    ckpt_path = argv[1]
    output_dir = Path(argv[2])
    backbone_name = argv[3]

    model = AutoModel.from_pretrained(backbone_name, trust_remote_code=True)
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

    pooling = Pooling(
        word_embedding_dimension=embedding_dim,
        pooling_mode_cls_token=True,
        pooling_mode_mean_tokens=False,
        pooling_mode_lasttoken=False,
        include_prompt=True,
    )
    hf_backbone[1] = pooling

    print(hf_backbone)
    print(hf_backbone.get_sentence_embedding_dimension())

    print(hf_backbone.encode(example))
    print(hf_backbone.encode(example).shape)

    hf_backbone.save_pretrained(str(output_dir / model_name))
