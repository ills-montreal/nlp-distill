from typing import List, Tuple, Dict

from torch.utils.data import Dataset
import torch
from pathlib import Path

import json
import numpy as np
from collections import defaultdict
import re


class TeacherEmbeddingDataset(Dataset):
    def __init__(self, teacher_path: Path):
        """
        :param teacher_path: Path to a directory containing teacher [name].npy
        and a jsonl files with the inputs with the same name
        """

        # teacher files paths, get all the .npy files
        self.teacher_path = teacher_path

        pattern = re.compile(r"embeddings-(\d+)-(\d+).npy")
        embedding_files = list(teacher_path.rglob("*/embeddings*"))

        # keep those that match the pattern
        embedding_files = [
            file for file in embedding_files if pattern.search(str(file))
        ]

        # keep only files up until 1000000
        embedding_files = [
            file
            for file in embedding_files
            if int(pattern.search(str(file)).group(2)) <= 10000000
        ]

        input_files = [
            Path(str(file).replace("embeddings-", "inputs-").replace("npy", "jsonl"))
            for file in embedding_files
        ]

        # Load the inputs
        self.inputs = []
        self.embeddings = []
        self.embedding_files = []
        self.input_files = []

        for input_file, file in zip(input_files, embedding_files):
            try:
                with open(input_file, "r") as f:
                    inputs = [json.loads(line) for line in f]
                with open(file, "rb") as f:
                    embeddings = np.load(f).astype(np.float32)

            except Exception as e:
                print(f"Error with {input_file}, {file}: {e}")
                continue
            else:
                self.inputs.extend(inputs)
                self.embeddings.append(embeddings)
                self.embedding_files.append(file)
                self.input_files.append(input_file)

        # Load the embeddings

        self.embeddings = np.concatenate(self.embeddings)
        self.embeddings -= self.embeddings.mean(axis=0)
        self.embeddings /= self.embeddings.std(axis=0)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx) -> Tuple[np.array, str]:
        return self.embeddings[idx], self.inputs[idx]

    def __repr__(self):
        return (
            f"{self.teacher_path} with {len(self)} samples"
            + "\n"
            + f"{self.embeddings.shape}, {len(self.inputs)}"
        )


class MultiTeacherEmbeddingDataset(Dataset):
    def __init__(self, teachers_path):
        """
        :param teachers_path: Path to a directory containing teacher directories
        """

        self.teacher_paths = teachers_path

        self.datasets = []
        for teacher in self.teacher_paths:
            try:
                ds = TeacherEmbeddingDataset(teacher)
                self.datasets.append(ds)
            except Exception as e:
                print(f"Error loading {teacher}: {e}")

        self.lengths = [len(dataset) for dataset in self.datasets]
        self.total_length = sum(self.lengths)

        self.cumulative_lengths = [0] + [
            sum(self.lengths[: i + 1]) for i in range(len(self.lengths))
        ]

        teacher_idx = 0
        self.idx2teacher = []
        for i in range(self.total_length):
            if i >= self.cumulative_lengths[teacher_idx + 1]:
                teacher_idx += 1
            self.idx2teacher.append(teacher_idx)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx) -> Tuple[np.array, str, int]:
        teacher_idx = self.idx2teacher[idx]
        return self.datasets[teacher_idx][
            idx - self.cumulative_lengths[teacher_idx]
        ] + (teacher_idx,)

    def __repr__(self):
        s = ""
        for i, dataset in enumerate(self.datasets):
            s += f"Teacher {i}: {dataset}\n"

        return s


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        """
        :param batch:
        :return: 3 lists of targets, inputs, teacher_idx. targets and inputs are lists
         of length n_teachers and contain the targets and inputs for each teacher.
        """
        batch = sorted(batch, key=lambda x: x[2])
        targets, inputs, teacher_idx = zip(*batch)

        targets = [torch.tensor(t) for t in targets]

        new_targets = []
        new_inputs = []
        new_teacher_indices = []

        curr_idx = 0
        curr_teacher_idx = teacher_idx[0]

        for k, teacher_idx in enumerate(teacher_idx):
            if teacher_idx != curr_teacher_idx:
                new_targets.append(torch.stack(targets[curr_idx:k]))
                n_input = tokenizer(
                    inputs[curr_idx:k],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                new_inputs.append(n_input)

                new_teacher_indices.append(curr_teacher_idx)
                curr_idx = k
                curr_teacher_idx = teacher_idx

        new_targets.append(torch.stack(targets[curr_idx:]))
        n_input = tokenizer(
            inputs[curr_idx:], return_tensors="pt", padding=True, truncation=True
        )
        new_inputs.append(n_input)
        new_teacher_indices.append(curr_teacher_idx)

        return new_targets, new_inputs, new_teacher_indices

    return collate_fn


class MultiTeacherAlignedEmbeddingDataset(Dataset):

    def __init__(self, teachers_path):
        self.teacher_paths = teachers_path

        self.datasets = []
        for teacher in self.teacher_paths:
            try:
                ds = TeacherEmbeddingDataset(teacher)
                self.datasets.append(ds)
            except Exception as e:
                print(f"Error loading {teacher}: {e}")

        text_to_embedding: Dict[str, List[Tuple[np.array, int]]] = defaultdict(list)

        for i, dataset in enumerate(self.datasets):
            for embedding, text in dataset:
                text_to_embedding[text].append((embedding, i))

        # sort the embeddings by teacher index
        for text, embeddings in text_to_embedding.items():
            text_to_embedding[text] = sorted(embeddings, key=lambda x: x[1])

        # make a list of aligned embeddings
        self.aligned_embeddings = []
        self.texts = []
        self.teacher_indices = []
        for text, embeddings in text_to_embedding.items():
            self.texts.append(text)
            self.aligned_embeddings.append([emb for emb, _ in embeddings])
            self.teacher_indices.append([idx for _, idx in embeddings])

        self.n_teachers = len(self.datasets)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Tuple[List[np.array], str, List[int]]:
        return self.aligned_embeddings[idx], self.texts[idx], self.teacher_indices[idx]

    def __repr__(self):
        s = ""
        for i, dataset in enumerate(self.datasets):
            s += f"Teacher {i}: {dataset}\n"

        s += f"Aligned dataset with {len(self)} samples"

        return s


def make_aligned_collate_fn(tokenizer, teachers_dims: List[int]):
    def collate_fn(batch):
        """

        :param batch:
        :return: 3 lists of targets, inputs, teacher_idx. targets and inputs are lists of
        length n_teachers and contain the targets and inputs for each teacher.
        """

        # targets is List[List[np.array]], inputs is List[str], teacher_idx is List[int]
        targets, inputs, teacher_idx = zip(*batch)

        mask = torch.zeros(len(teachers_dims), len(inputs), dtype=torch.bool)

        new_input = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        targets = []
        for teacher in range(len(teachers_dims)):
            target = torch.zeros(
                len(inputs), teachers_dims[teacher], dtype=torch.float32
            )
            targets.append(target)

        for i, (aligned_embeddings, _, teacher_indices) in enumerate(batch):
            for embedding, teacher_idx in zip(aligned_embeddings, teacher_indices):
                targets[teacher_idx][i] = torch.tensor(embedding)
                mask[teacher_idx, i] = True

        return targets, new_input, mask

    return collate_fn
