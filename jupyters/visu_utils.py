import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import json

from IPython.core.display_functions import display
from nbconvert.filters import escape_latex
from pandas import pivot

TASK_LIST_CLASSIFICATION = [
    "AmazonCounterfactualClassification",
    "AmazonPolarityClassification",
    "AmazonReviewsClassification",
    "Banking77Classification",
    "EmotionClassification",
    "ImdbClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MTOPDomainClassification",
    "MTOPIntentClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",
    "ArxivClusteringS2S",
    "BiorxivClusteringP2P",
    "BiorxivClusteringS2S",
    "MedrxivClusteringP2P",
    "MedrxivClusteringS2S",
    "RedditClustering",
    "RedditClusteringP2P",
    "StackExchangeClustering",
    "StackExchangeClusteringP2P",
    "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    "AskUbuntuDupQuestions",
    "MindSmallReranking",
    "SciDocsRR",
    "StackOverflowDupQuestions",
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    "ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_STS = [
    "BIOSSES",
    "SICK-R",
    "STS12",
    "STS13",
    "STS14",
    "STS15",
    "STS16",
    "STS17",
    "STS22",
    "STSBenchmark",
    "SummEval",
]


def highlight_top_1(s, props=""):
    max_value = s.max()
    is_max = s == max_value
    return [props if v else "" for v in is_max]


def highlight_top_2(s, props=""):
    max_values = s.nlargest(2)
    is_max = s.isin(max_values)
    return [props if v else "" for v in is_max]


def highlight_top_3(s, props=""):
    max_values = s.nlargest(3)
    is_max = s.isin(max_values)
    return [props if v else "" for v in is_max]


def make_results_tables_per_group_size(
    df, metric, low, high, EXPORT_PATH_TABLE, name, caption
):
    df_pivot = df.pivot_table(
        index=["loss", "Model", "Model Size (Million Parameters)", "Dataset"],
        columns="Task",
        values=metric,
    ).reset_index()

    # replace " " by \\

    df_pivot = df_pivot.reset_index().rename(
        {
            "loss": " ",
            "Model Size (Million Parameters)": "Params. (M)",
            "Dataset": "  ",
        },
        axis=1,
    )
    df_pivot = df_pivot.drop(columns=["  "])
    # drop any columns with index
    df_pivot = df_pivot.drop(
        columns=[c for c in df_pivot.columns if c.startswith("level") or c == "index"]
    )
    df_pivot = df_pivot.dropna()

    df_pivot["Model"] = df_pivot["Model"].apply(lambda x: x.split("/")[-1])

    latex_results = df_pivot.set_index([" ", "Model"])

    params = latex_results["Params. (M)"]
    latex_results = latex_results.drop(columns=["Params. (M)"])

    avg_columns = latex_results.mean(axis=1)

    latex_results["Avg."] = avg_columns

    columns = latex_results.columns
    latex_results["Size"] = params
    df_pivot = latex_results[["Size"] + list(columns)]

    df_pivot = df_pivot.drop_duplicates()
    df_pivot = df_pivot[~df_pivot.index.duplicated(keep="first")]

    df_pivot = df_pivot.loc[["MTEB", "MSE", "NLL"]]

    latex_results = df_pivot.style.format("{:.1f}")
    # format params as int
    latex_results = latex_results.format("{:.0f}M", subset=["Size"])
    latex_results = latex_results.format_index(escape="latex")

    # highlight max
    # latex_results = latex_results.highlight_max(axis=0, props="bfseries:")

    # apply to all but the first column
    latex_results = latex_results.apply(
        lambda x: highlight_top_1(x, "bfseries:"),
        axis=0,
        subset=latex_results.columns[1:],
    )
    # underline top 2
    latex_results = latex_results.apply(
        lambda x: highlight_top_2(x, "underline:--rwrap"),
        axis=0,
        subset=latex_results.columns[1:],
    )

    latex = latex_results.to_latex(
        clines="skip-last;data",
        hrules=True,
        sparse_index=True,
        multicol_align="c",
        multirow_align="c",
        caption=caption,
        label=f"tab:{name}",
        column_format="llc|" + "c" * (len(df_pivot.columns) - 2) + "|c",
    )

    # add resizebox
    latex = latex.replace(
        r"\begin{tabular}", r"\resizebox{\textwidth}{!}{\begin{tabular}"
    )
    latex = latex.replace(r"\end{tabular}", r"\end{tabular}}")

    with open(EXPORT_PATH_TABLE / f"{name}.tex", "w") as f:
        f.write(latex)


def load_classification_merged_mteb(MTEB_BASELINES_PATH, results_path):
    df_mteb_classifications = pd.read_csv(MTEB_BASELINES_PATH)
    df_mteb_classifications

    def extract_url_from_html_link(html):
        return re.findall(r'href=[\'"]?([^\'" >]+)', html)[0]

    df_mteb_classifications["Model"] = df_mteb_classifications["Model"].apply(
        extract_url_from_html_link
    )
    df_mteb_classifications["Model"] = df_mteb_classifications["Model"].apply(
        lambda x: "/".join(x.split("/")[-2:])
    )

    # Remove models with "test" in the name
    df_mteb_classifications = df_mteb_classifications[
        ~df_mteb_classifications["Model"].str.contains("test")
    ]

    df_mteb_classifications["loss"] = "MTEB"
    df_mteb_classifications["Dataset"] = "MTEB"
    df_mteb_classifications["Training step"] = 0

    results_df = []
    for path in results_path:
        df, df_ = load_ours_classification(path)

        results_df.append(df_)

    df_merged = pd.concat(results_df + [df_mteb_classifications], ignore_index=True)[
        results_df[0].columns
    ]

    df_merged = df_merged[~df_merged["Model"].str.contains("glove.6B")]

    return df_merged


def load_ours_classification(MTEB_PATHS):
    OURS_SIZES = {"xs": 23, "s": 33, "m": 109, "l": 335}

    # list all json
    json_files = list(MTEB_PATHS.rglob("*.json"))

    # load all json
    data = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            data.append(json.load(f))

    records = []

    for path, results in zip(json_files, data):
        model_ref = path.parts[-4]
        expe_name = path.parts[-5]
        expe = path.parts[-7]
        step = int(model_ref.split("=")[-1])

        if path.stem == "model_meta":
            continue

        dataset = "BIG" if not "gist" in expe else "GIST"
        loss = "NLL" if "nll" in expe else "MSE"
        if "single" in expe:
            loss += "-Single"

        model_size_str = expe_name.split("-")[-1]

        model = f"{loss}/{dataset}/Student-{model_size_str}"

        main_score = results["scores"]["test"][0]["main_score"]
        task_name = results["task_name"]

        if task_name not in TASK_LIST_CLASSIFICATION:
            continue
        records.append(
            {
                "Model": model,
                "Training step": step,
                "loss": loss,
                "Dataset": dataset,
                "Task": task_name,
                "Accuracy": main_score * 100,
                "Model Size (Million Parameters)": OURS_SIZES[expe_name.split("-")[-1]],
            }
        )

    df = pd.DataFrame(records)

    # Index(['model', 'Task', 'Accuracy', 'Model Size (Million Parameters)'], dtype='object')

    # make columns with Task content filled with Accuracy
    df_ = df.pivot(
        index=[
            "Model",
            "Model Size (Million Parameters)",
            "loss",
            "Training step",
            "Dataset",
        ],
        columns="Task",
        values="Accuracy",
    )
    # add " (en)" to the column names

    # source columns
    # Index(['AmazonCounterfactualClassification', 'AmazonPolarityClassification',
    #        'AmazonReviewsClassification', 'Banking77Classification',
    #        'EmotionClassification', 'ImdbClassification',
    #        'MTOPDomainClassification', 'MTOPIntentClassification',
    #        'MassiveIntentClassification', 'MassiveScenarioClassification',
    #        'ToxicConversationsClassification',
    #        'TweetSentimentExtractionClassification'],
    #       dtype='object', name='Task')

    # Target columns
    # Index(['Rank', 'Model', 'Model Size (Million Parameters)',
    #        'Memory Usage (GB, fp32)', 'Average',
    #        'AmazonCounterfactualClassification (en)',
    #        'AmazonPolarityClassification', 'AmazonReviewsClassification (en)',
    #        'Banking77Classification', 'EmotionClassification',
    #        'ImdbClassification', 'MassiveIntentClassification (en)',
    #        'MassiveScenarioClassification (en)', 'MTOPDomainClassification (en)',
    #        'MTOPIntentClassification (en)', 'ToxicConversationsClassification',
    #        'TweetSentimentExtractionClassification', 'Subset', 'Task category'],
    #       dtype='object')

    COLUMN_MAPPING = {
        "AmazonCounterfactualClassification": "AmazonCounterfactualClassification (en)",
        "AmazonReviewsClassification": "AmazonReviewsClassification (en)",
        "MassiveIntentClassification": "MassiveIntentClassification (en)",
        "MassiveScenarioClassification": "MassiveScenarioClassification (en)",
        "MTOPDomainClassification": "MTOPDomainClassification (en)",
        "MTOPIntentClassification": "MTOPIntentClassification (en)",
    }

    df_ = df_.rename(columns=COLUMN_MAPPING)

    df_ = df_.reset_index()

    return df, df_


def load_ours_clustering(MTEB_PATHS):
    OURS_SIZES = {"xs": 23, "s": 33, "m": 109, "l": 335}

    # list all json
    json_files = list(MTEB_PATHS.rglob("*.json"))

    # load all json
    data = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            data.append(json.load(f))

    records = []

    for path, results in zip(json_files, data):
        model_ref = path.parts[-4]
        expe_name = path.parts[-5]
        expe = path.parts[-7]
        step = int(model_ref.split("=")[-1])

        if path.stem == "model_meta":
            continue

        dataset = "BIG" if not "gist" in expe else "GIST"
        loss = "NLL" if "nll" in expe else "MSE"
        if "single" in expe:
            loss += "-Single"

        model_size_str = expe_name.split("-")[-1]

        model = f"{loss}/{dataset}/Student-{model_size_str}"

        main_score = results["scores"]["test"][0]["main_score"]
        task_name = results["task_name"]

        if task_name not in TASK_LIST_CLUSTERING:
            continue
        records.append(
            {
                "Model": model,
                "Training step": step,
                "loss": loss,
                "Dataset": dataset,
                "Task": task_name,
                "Score": main_score * 100,
                "Model Size (Million Parameters)": OURS_SIZES[expe_name.split("-")[-1]],
            }
        )

    df = pd.DataFrame(records)

    # Index(['model', 'Task', 'Accuracy', 'Model Size (Million Parameters)'], dtype='object')

    # make columns with Task content filled with Accuracy
    df_ = df.pivot(
        index=[
            "Model",
            "Model Size (Million Parameters)",
            "loss",
            "Training step",
            "Dataset",
        ],
        columns="Task",
        values="Score",
    )

    # df_ = df_.rename(columns=COLUMN_MAPPING)

    df_ = df_.reset_index()

    return df, df_


def load_clustering_merged_mteb(MTEB_BASELINES_PATH, results_path):
    df_mteb_classifications = pd.read_csv(MTEB_BASELINES_PATH)
    df_mteb_classifications

    def extract_url_from_html_link(html):
        return re.findall(r'href=[\'"]?([^\'" >]+)', html)[0]

    df_mteb_classifications["Model"] = df_mteb_classifications["Model"].apply(
        extract_url_from_html_link
    )
    df_mteb_classifications["Model"] = df_mteb_classifications["Model"].apply(
        lambda x: "/".join(x.split("/")[-2:])
    )

    # Remove models with "test" in the name
    df_mteb_classifications = df_mteb_classifications[
        ~df_mteb_classifications["Model"].str.contains("test")
    ]

    df_mteb_classifications["loss"] = "MTEB"
    df_mteb_classifications["Dataset"] = "MTEB"
    df_mteb_classifications["Training step"] = 0

    df_mteb_classifications = df_mteb_classifications.dropna()

    results_df = []
    for path in results_path:
        df, df_ = load_ours_clustering(path)

        results_df.append(df_)

    df_merged = pd.concat(results_df + [df_mteb_classifications], ignore_index=True)[
        results_df[0].columns
    ]

    df_merged = df_merged[~df_merged["Model"].str.contains("glove.6B")]

    return df_merged


def load_ours_sts(MTEB_PATHS):
    OURS_SIZES = {"xs": 23, "s": 33, "m": 109, "l": 335}

    # list all json
    json_files = list(MTEB_PATHS.rglob("*.json"))

    # load all json
    data = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            data.append(json.load(f))

    records = []

    for path, results in zip(json_files, data):
        model_ref = path.parts[-4]
        expe_name = path.parts[-5]
        expe = path.parts[-7]
        step = int(model_ref.split("=")[-1])

        if path.stem == "model_meta":
            continue

        dataset = "BIG" if not "gist" in expe else "GIST"
        loss = "NLL" if "nll" in expe else "MSE"
        if "single" in expe:
            loss += "-Single"

        model_size_str = expe_name.split("-")[-1]

        model = f"{loss}/{dataset}/Student-{model_size_str}"

        main_score = results["scores"]["test"][0]["main_score"]
        task_name = results["task_name"]

        if task_name not in TASK_LIST_STS:
            continue
        records.append(
            {
                "Model": model,
                "Training step": step,
                "loss": loss,
                "Dataset": dataset,
                "Task": task_name,
                "Score": main_score * 100,
                "Model Size (Million Parameters)": OURS_SIZES[expe_name.split("-")[-1]],
            }
        )

    df = pd.DataFrame(records)

    # Index(['model', 'Task', 'Accuracy', 'Model Size (Million Parameters)'], dtype='object')

    # make columns with Task content filled with Accuracy
    df_ = df.pivot(
        index=[
            "Model",
            "Model Size (Million Parameters)",
            "loss",
            "Training step",
            "Dataset",
        ],
        columns="Task",
        values="Score",
    )

    # df_ = df_.rename(columns=COLUMN_MAPPING)

    df_ = df_.reset_index()

    return df, df_


def load_sts_merged_mteb(MTEB_BASELINES_PATH, results_path):
    df_mteb_classifications = pd.read_csv(MTEB_BASELINES_PATH)
    df_mteb_classifications

    def extract_url_from_html_link(html):
        return re.findall(r'href=[\'"]?([^\'" >]+)', html)[0]

    df_mteb_classifications["Model"] = df_mteb_classifications["Model"].apply(
        extract_url_from_html_link
    )
    df_mteb_classifications["Model"] = df_mteb_classifications["Model"].apply(
        lambda x: "/".join(x.split("/")[-2:])
    )

    # Remove models with "test" in the name
    df_mteb_classifications = df_mteb_classifications[
        ~df_mteb_classifications["Model"].str.contains("test")
    ]

    df_mteb_classifications["loss"] = "MTEB"
    df_mteb_classifications["Dataset"] = "MTEB"
    df_mteb_classifications["Training step"] = 0

    df_mteb_classifications = df_mteb_classifications.dropna()
    # remove " (en)"
    df_mteb_classifications.columns = [
        c.replace(" (en)", "") for c in df_mteb_classifications.columns
    ]

    df_mteb_classifications.columns = [
        c.replace(" (en-en)", "") for c in df_mteb_classifications.columns
    ]

    results_df = []
    for path in results_path:
        df, df_ = load_ours_sts(path)

        results_df.append(df_)

    df_merged = pd.concat(results_df + [df_mteb_classifications], ignore_index=True)[
        results_df[0].columns
    ]

    df_merged = df_merged[~df_merged["Model"].str.contains("glove.6B")]

    return df_merged


def load_ours_retrieval(MTEB_PATHS):
    OURS_SIZES = {"xs": 23, "s": 33, "m": 109, "l": 335}

    # list all json
    json_files = list(MTEB_PATHS.rglob("*.json"))

    # load all json
    data = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            data.append(json.load(f))

    records = []

    for path, results in zip(json_files, data):
        model_ref = path.parts[-4]
        expe_name = path.parts[-5]
        expe = path.parts[-7]
        step = int(model_ref.split("=")[-1])

        if path.stem == "model_meta":
            continue

        dataset = "BIG" if not "gist" in expe else "GIST"
        loss = "NLL" if "nll" in expe else "MSE"
        if "single" in expe:
            loss += "-Single"

        model_size_str = expe_name.split("-")[-1]

        model = f"{loss}/{dataset}/Student-{model_size_str}"

        main_score = results["scores"]["test"][0]["main_score"]
        task_name = results["task_name"]

        if task_name not in TASK_LIST_RETRIEVAL:
            continue
        records.append(
            {
                "Model": model,
                "Training step": step,
                "loss": loss,
                "Dataset": dataset,
                "Task": task_name,
                "Score": main_score * 100,
                "Model Size (Million Parameters)": OURS_SIZES[expe_name.split("-")[-1]],
            }
        )

    df = pd.DataFrame(records)

    # Index(['model', 'Task', 'Accuracy', 'Model Size (Million Parameters)'], dtype='object')

    # make columns with Task content filled with Accuracy
    df_ = df.pivot(
        index=[
            "Model",
            "Model Size (Million Parameters)",
            "loss",
            "Training step",
            "Dataset",
        ],
        columns="Task",
        values="Score",
    )

    # df_ = df_.rename(columns=COLUMN_MAPPING)

    df_ = df_.reset_index()

    return df, df_


def load_retrieval_merged_mteb(MTEB_BASELINES_PATH, results_path):
    df_mteb_classifications = pd.read_csv(MTEB_BASELINES_PATH)
    df_mteb_classifications

    def extract_url_from_html_link(html):
        return re.findall(r'href=[\'"]?([^\'" >]+)', html)[0]

    df_mteb_classifications["Model"] = df_mteb_classifications["Model"].apply(
        extract_url_from_html_link
    )
    df_mteb_classifications["Model"] = df_mteb_classifications["Model"].apply(
        lambda x: "/".join(x.split("/")[-2:])
    )

    # Remove models with "test" in the name
    df_mteb_classifications = df_mteb_classifications[
        ~df_mteb_classifications["Model"].str.contains("test")
    ]

    df_mteb_classifications["loss"] = "MTEB"
    df_mteb_classifications["Dataset"] = "MTEB"
    df_mteb_classifications["Training step"] = 0

    df_mteb_classifications = df_mteb_classifications.dropna()
    # remove " (en)"
    df_mteb_classifications.columns = [
        c.replace(" (en)", "") for c in df_mteb_classifications.columns
    ]

    df_mteb_classifications.columns = [
        c.replace(" (en-en)", "") for c in df_mteb_classifications.columns
    ]

    results_df = []
    for path in results_path:
        df, df_ = load_ours_retrieval(path)

        results_df.append(df_)

    df_merged = pd.concat(results_df + [df_mteb_classifications], ignore_index=True)[
        results_df[0].columns
    ]

    df_merged = df_merged[~df_merged["Model"].str.contains("glove.6B")]

    return df_merged


def load_ours_pairsclassification(MTEB_PATHS):
    OURS_SIZES = {"xs": 23, "s": 33, "m": 109, "l": 335}

    # list all json
    json_files = list(MTEB_PATHS.rglob("*.json"))

    # load all json
    data = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            data.append(json.load(f))

    records = []

    for path, results in zip(json_files, data):
        model_ref = path.parts[-4]
        expe_name = path.parts[-5]
        expe = path.parts[-7]
        step = int(model_ref.split("=")[-1])

        if path.stem == "model_meta":
            continue

        dataset = "BIG" if not "gist" in expe else "GIST"
        loss = "NLL" if "nll" in expe else "MSE"
        if "single" in expe:
            loss += "-Single"

        model_size_str = expe_name.split("-")[-1]

        model = f"{loss}/{dataset}/Student-{model_size_str}"

        main_score = results["scores"]["test"][0]["main_score"]
        task_name = results["task_name"]

        if task_name not in TASK_LIST_PAIR_CLASSIFICATION:
            continue
        records.append(
            {
                "Model": model,
                "Training step": step,
                "loss": loss,
                "Dataset": dataset,
                "Task": task_name,
                "Score": main_score * 100,
                "Model Size (Million Parameters)": OURS_SIZES[expe_name.split("-")[-1]],
            }
        )

    df = pd.DataFrame(records)

    # Index(['model', 'Task', 'Accuracy', 'Model Size (Million Parameters)'], dtype='object')

    # make columns with Task content filled with Accuracy
    df_ = df.pivot(
        index=[
            "Model",
            "Model Size (Million Parameters)",
            "loss",
            "Training step",
            "Dataset",
        ],
        columns="Task",
        values="Score",
    )

    # df_ = df_.rename(columns=COLUMN_MAPPING)

    df_ = df_.reset_index()

    return df, df_


def load_pairsclassification_merged_mteb(MTEB_BASELINES_PATH, results_path):
    df_mteb_classifications = pd.read_csv(MTEB_BASELINES_PATH)
    df_mteb_classifications

    def extract_url_from_html_link(html):
        return re.findall(r'href=[\'"]?([^\'" >]+)', html)[0]

    df_mteb_classifications["Model"] = df_mteb_classifications["Model"].apply(
        extract_url_from_html_link
    )
    df_mteb_classifications["Model"] = df_mteb_classifications["Model"].apply(
        lambda x: "/".join(x.split("/")[-2:])
    )

    # Remove models with "test" in the name
    df_mteb_classifications = df_mteb_classifications[
        ~df_mteb_classifications["Model"].str.contains("test")
    ]

    df_mteb_classifications["loss"] = "MTEB"
    df_mteb_classifications["Dataset"] = "MTEB"
    df_mteb_classifications["Training step"] = 0

    df_mteb_classifications = df_mteb_classifications.dropna()
    # remove " (en)"
    df_mteb_classifications.columns = [
        c.replace(" (en)", "") for c in df_mteb_classifications.columns
    ]

    df_mteb_classifications.columns = [
        c.replace(" (en-en)", "") for c in df_mteb_classifications.columns
    ]

    results_df = []
    for path in results_path:
        df, df_ = load_ours_retrieval(path)

        results_df.append(df_)

    df_merged = pd.concat(results_df + [df_mteb_classifications], ignore_index=True)[
        results_df[0].columns
    ]

    df_merged = df_merged[~df_merged["Model"].str.contains("glove.6B")]

    return df_merged
