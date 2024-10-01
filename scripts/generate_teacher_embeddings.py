import argparse
import json
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
import re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--flash_attn", action="store_true", default=False)
    parser.add_argument("--no_float16", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = load_dataset(args.dataset, split="train")

    try:
        model = SentenceTransformer(
            args.model,
            model_kwargs={
                "torch_dtype": torch.float16 if not args.no_float16 else torch.float32,
                "attn_implementation": (
                    "flash_attention_2" if args.flash_attn else None
                ),
            },
            device="cuda",
            trust_remote_code=True,
        )
    except Exception as e:
        model = SentenceTransformer(
            args.model,
            model_kwargs={
                "torch_dtype": torch.float16 if not args.no_float16 else torch.float32,
            },
            device="cuda",
            trust_remote_code=True,
        )

    if model.tokenizer.eos_token is not None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    output_dir = Path(args.output_dir) / args.model / args.dataset / "train"
    output_dir.mkdir(parents=True, exist_ok=True)

    # find all files that fit the pattern
    # f"embeddings-{i}-{i+100000}.npy"
    # make regex pattern to retrieve start and end
    pattern = re.compile(r"embeddings-(\d+)-(\d+).npy")
    # find all files that fit the pattern
    files = list(output_dir.glob("embeddings-*.npy"))

    start = args.start
    if len(files) > 0:
        # extract start and end from the file names
        ends = [int(pattern.search(str(file)).group(2)) for file in files]
        # get largest end
        start = max(ends)

    # split dataset["text"] into chunk of 100000
    for i in range(start, len(dataset["text"]), 100000):
        texts = dataset["text"][i : i + 100000]
        embeddings = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        embeddings = embeddings.astype(np.float16)

        with open(output_dir / f"embeddings-{i}-{i+100000}.npy", "wb") as f:
            np.save(f, embeddings)

        # save text in jsonl
        with open(output_dir / f"inputs-{i}-{i+100000}.jsonl", "w") as f:
            for text in texts:
                f.write(json.dumps(text) + "\n")


if __name__ == "__main__":
    import sys

    print(sys.version)
    main()
