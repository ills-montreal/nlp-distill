from datasets import load_dataset, Dataset
import pandas as pd


def main():

    dataset = load_dataset("avsolatorio/medi-data-mteb_avs_triplets")

    dataset = dataset["train"]

    samples = []
    for sample in dataset:
        query, pos, neg = sample["query"], sample["pos"], sample["neg"]
        samples.append(query)
        samples.append(pos)
        samples.append(neg)

    df = pd.DataFrame(samples, columns=["text"])
    df = df.drop_duplicates()

    dataset = Dataset.from_pandas(df)

    dataset.push_to_hub("Icannos/distillation_training_gist_medi_mteb")
    load_dataset("Icannos/distillation_training_gist_medi_mteb")


if __name__ == "__main__":
    main()
