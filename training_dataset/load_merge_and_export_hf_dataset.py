from datasets import load_dataset, Dataset
import pandas as pd


"""
The goal of this script is to load the different embedding dataset from huggingface
and to flatten them into a single dataset containing only "text".
"""


def load_specter():
    """
    embedding-data/SPECTER
    :return:
    """
    dataset = load_dataset("embedding-data/specter")
    dataset = dataset["train"]

    samples = []

    for elem in dataset:
        s1, s2 = elem["set"][0], elem["set"][1]
        samples.append(s1)
        samples.append(s2)

    df = pd.DataFrame(samples, columns=["text"])
    df = df.drop_duplicates()

    return df


def load_amazon_qa():
    dataset = load_dataset("embedding-data/Amazon-QA")
    dataset = dataset["train"]

    samples = []
    for elem in dataset:
        query = elem["query"]
        samples.append(query)
        for answer in elem["pos"]:
            samples.append(answer)

    df = pd.DataFrame(samples, columns=["text"])
    df = df.drop_duplicates()

    return df


def load_simple_wiki():
    dataset = load_dataset("embedding-data/simple-wiki")
    dataset = dataset["train"]

    samples = []
    for elem in dataset:
        for s in elem["set"]:
            samples.append(s)

    df = pd.DataFrame(samples, columns=["text"])
    df = df.drop_duplicates()

    return df


def load_QQP_triplets():
    dataset = load_dataset("embedding-data/QQP_triplets")
    dataset = dataset["train"]

    samples = []

    for elem in dataset:
        d = elem["set"]
        samples.append(d["query"])

        for s in d["pos"]:
            samples.append(s)

        for s in d["neg"]:
            samples.append(s)

    df = pd.DataFrame(samples, columns=["text"])
    df = df.drop_duplicates()

    return df


def load_sentence_compression():
    dataset = load_dataset("embedding-data/sentence-compression")
    dataset = dataset["train"]

    samples = []
    for elem in dataset:
        for s in elem["set"]:
            samples.append(s)

    df = pd.DataFrame(samples, columns=["text"])
    df = df.drop_duplicates()

    return df


def load_altlex():
    dataset = load_dataset("embedding-data/altlex")
    dataset = dataset["train"]

    samples = []
    for elem in dataset:
        for s in elem["set"]:
            samples.append(s)

    df = pd.DataFrame(samples, columns=["text"])
    df = df.drop_duplicates()

    return df


def load_agnews():
    dataset = load_dataset("fancyzhx/ag_news")
    dataset = dataset["train"]

    df = dataset.to_pandas()
    df = df["text"].to_frame()

    return df


def load_sst2():
    dataset = load_dataset("stanfordnlp/sst2")
    dataset = dataset["train"]
    df = dataset.to_pandas()["sentence"].to_frame()
    df = df.rename(columns={"sentence": "text"})

    return df


def load_dair_emotion():
    dataset = load_dataset("dair-ai/emotion", "unsplit")
    dataset = dataset["train"]
    df = dataset.to_pandas()
    df = df["text"].to_frame()

    return df


def load_snli():
    dataset = load_dataset("stanfordnlp/snli")

    dataset = dataset["train"]

    premise, hypothesis = (dataset["premise"], dataset["hypothesis"])
    df_premise = pd.DataFrame(premise, columns=["text"])
    df_hypothesis = pd.DataFrame(hypothesis, columns=["text"])

    df = pd.concat([df_premise, df_hypothesis])

    return df


def tweet_eval():
    dataset = load_dataset("cardiffnlp/tweet_eval", "emoji")

    dataset = dataset["train"]

    df = dataset.to_pandas()
    df = df["text"].to_frame()

    return df


def load_imdb():
    dataset = load_dataset("stanfordnlp/imdb")
    dataset = dataset["train"]

    df = dataset.to_pandas()
    df = df["text"].to_frame()

    return df


def main():

    datasets = [
        load_specter(),
        load_amazon_qa(),
        load_simple_wiki(),
        load_QQP_triplets(),
        load_sentence_compression(),
        load_altlex(),
        load_agnews(),
        load_sst2(),
        load_dair_emotion(),
        load_snli(),
        tweet_eval(),
        load_imdb(),
    ]

    df = pd.concat(datasets)

    df = df.drop_duplicates()
    df = df["text"].to_frame()

    # make a dataset from the pandas dataframe
    dataset = Dataset.from_pandas(df)

    dataset.push_to_hub("Icannos/distillation_training_1")


if __name__ == "__main__":
    main()
