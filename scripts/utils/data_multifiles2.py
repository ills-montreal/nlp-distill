import json
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiTeacherAlignedEmbeddingDataset(Dataset):
    def __init__(self, teachers_paths: List[Path], max_samples: int = -1):
        self.max_samples = max_samples

        (
            self.idx_to_file_and_pos,  # For each text, a list of size N_teacher, for each the idx in in_teachers_locations
            self.in_teachers_locations,
            self.teachers_locations,
        ) = self.__get_all_texts_and_files_refs(teachers_paths)

        print("================")
        print(
            "max indices",
            self.idx_to_file_and_pos.max(0),
            self.idx_to_file_and_pos.max(0).shape,
        )
        print(
            "max indices",
            self.idx_to_file_and_pos.max(1),
            self.idx_to_file_and_pos.max(1).shape,
        )
        print(self.idx_to_file_and_pos.shape)

        print("================")
        print(
            f"in_teacher_locations: {[(teacher_idx,t.shape) for teacher_idx, t in enumerate(self.in_teachers_locations)]}"
        )

        self.cached_teacher_files_indices = [-1] * len(teachers_paths)
        self.cached_embeddings_files: List[Optional[np.array]] = [None] * len(
            teachers_paths
        )
        self.cached_inputs_files: List[Optional[List[str]]] = [None] * len(
            teachers_paths
        )

        _ = self.__getitem__(0)  # cache the first element

    def __len__(self):
        return self.idx_to_file_and_pos.shape[0]

    def roll(self, worker_id=-1, n_workers=-1):
        # rotate the texts so each worker has a different starting point
        if worker_id != -1 and n_workers != -1:
            N = len(self.idx_to_file_and_pos)
            roll_size = N // n_workers
            shift = worker_id * roll_size
            self.idx_to_file_and_pos = np.roll(
                self.idx_to_file_and_pos, shift=shift, axis=0
            )

    def teachers_embeddings_dims(self):
        return [emb.shape[-1] for emb in self.cached_embeddings_files]

    def __getitem__(self, item) -> Tuple[List[np.array], str, List[int]]:

        embeddings = []
        teacher_indices = []

        texts = []

        for teacher_idx, in_teacher_idx in enumerate(self.idx_to_file_and_pos[item]):
            if in_teacher_idx == -1:
                continue

            t_idx, in_file_idx, file_idx = self.in_teachers_locations[teacher_idx][
                in_teacher_idx
            ]

            text, embedding = self.__get_text_and_embedding(
                teacher_idx, file_idx, in_file_idx
            )
            embeddings.append(embedding)
            teacher_indices.append(teacher_idx)
            texts.append(text)

        # check that all the texts are the same
        assert all(
            [t == texts[0] for t in texts]
        ), "Texts are not the same across teachers"

        if len(texts) <= 0:
            print(self.idx_to_file_and_pos[item])
            raise ValueError(f"No embeddings found for text {item}")

        return embeddings, texts[0], teacher_indices

    def __get_text_and_embedding(self, teacher_idx, file_idx, in_file_idx):
        cached_file_idx = self.cached_teacher_files_indices[teacher_idx]

        if cached_file_idx != file_idx:
            inputs_path, embedding_path = self.teachers_locations[teacher_idx][
                file_idx
            ][2:]

            with open(inputs_path, "r") as f:
                inputs = [json.loads(line) for line in f]

            embedding = np.load(embedding_path)

            self.cached_teacher_files_indices[teacher_idx] = file_idx

            self.cached_embeddings_files[teacher_idx] = embedding
            self.cached_inputs_files[teacher_idx] = inputs

        text, embedding = (
            self.cached_inputs_files[teacher_idx][in_file_idx],
            self.cached_embeddings_files[teacher_idx][in_file_idx],
        )

        return text, embedding

    def __get_all_texts_and_files_refs(self, teachers_paths: List[Path]):
        texts: Dict[str, Dict] = defaultdict(dict)

        teachers_locations = []
        N_teachers = len(teachers_paths)

        for teacher_idx, teacher_path in enumerate(teachers_paths):
            teachers_locations_ = []
            for files in zip(*self.__extract_teacher_files(teacher_path)):
                (start, end, input_file), (start_emb, end_emb, embedding_file) = files

                assert (
                    start == start_emb and end == end_emb
                ), f"Mismatched files {files}"
                teachers_locations_.append((start, end, input_file, embedding_file))
                with open(input_file, "r") as f:
                    inputs = [json.loads(line) for line in f]

                for in_file_idx, l in enumerate(inputs):
                    texts[l][teacher_idx] = (
                        in_file_idx,
                        len(teachers_locations_) - 1,  # file_idx
                    )

                if self.max_samples > 0 and end > self.max_samples:
                    break
            teachers_locations.append(teachers_locations_)

        assert N_teachers == len(
            teachers_locations
        ), f"Mismatched teacher files {N_teachers} != {len(teachers_locations)}"

        in_teacher_locations = [[] for _ in range(N_teachers)]
        idx_to_in_teacher_idx = []

        for t_idx, teachers in enumerate(texts.values()):
            # teachers : Dict[teacher_idx : int, (in_file_idx, file_idx)]
            in_teachers_indices: List[int] = [-1 for _ in range(N_teachers)]

            for teacher_idx, (in_file_idx, file_idx) in teachers.items():
                in_teacher_locations[teacher_idx].append([t_idx, in_file_idx, file_idx])
                in_teachers_indices[teacher_idx] = (
                    len(in_teacher_locations[teacher_idx]) - 1
                )

            idx_to_in_teacher_idx.append(in_teachers_indices)

        in_teacher_locations: List[np.array] = [
            np.array(l) for l in in_teacher_locations
        ]
        idx_to_in_teacher_idx: np.array = np.array(idx_to_in_teacher_idx)

        return idx_to_in_teacher_idx, in_teacher_locations, teachers_locations

    @staticmethod
    def __extract_teacher_files(teacher_path: Path):
        pattern = re.compile(r"embeddings-(\d+)-(\d+).npy")
        embedding_files = list(teacher_path.rglob("*/embeddings*"))

        _embedding_files = []
        _bounds = []
        for file in embedding_files:
            match = pattern.search(str(file))
            if match:
                _embedding_files.append(
                    (int(match.group(1)), int(match.group(2)), file)
                )

        embedding_files = _embedding_files

        input_files = [
            (
                start,
                end,
                Path(
                    str(file).replace("embeddings-", "inputs-").replace("npy", "jsonl")
                ),
            )
            for start, end, file in embedding_files
        ]

        return input_files, embedding_files

    def __repr__(self):
        txt = f"MultiTeacherAlignedEmbeddingDataset with {len(self)} samples\n"
        # list teachers and their files
        for teacher_idx, teacher_files in enumerate(self.teachers_locations):
            txt += "======================"
            txt += f"Teacher {teacher_idx}:\n"
            for start, end, input_file, embedding_file in teacher_files:
                txt += f"\t{input_file} -> {embedding_file}\n"

        for teacher_idx, dim in enumerate(self.teachers_embeddings_dims()):
            txt += f"Teacher {teacher_idx} has embeddings of dimension {dim}\n"

        return txt


class ExtractedSubSet(Dataset):
    def __init__(
        self, dataset: MultiTeacherAlignedEmbeddingDataset, indices: List[int]
    ):
        self.dataset = dataset

        self.datapoints = [dataset[i] for i in indices]

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, item):
        return self.datapoints[item]


def make_aligned_collate_fn(tokenizer, teachers_dims: List[int]):
    def collate_fn(batch):
        """

        :param batch:
        :return: 3 lists of targets, inputs, teacher_idx. targets and inputs are lists of
        length n_teachers and contain the targets and inputs for each teacher.
        """

        # targets is List[List[np.array]], inputs is List[str], teacher_idx is List[int]
        _, inputs, teacher_idx = zip(*batch)

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

        return targets, new_input, mask, inputs

    return collate_fn
