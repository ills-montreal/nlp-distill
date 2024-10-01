"""Example script for benchmarking all datasets constituting the MTEB English leaderboard & average scores"""

from __future__ import annotations

import subprocess
from pathlib import Path
from sys import argv

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

TASK_LIST = (
    TASK_LIST_CLASSIFICATION
    + TASK_LIST_CLUSTERING
    + TASK_LIST_PAIR_CLASSIFICATION
    + TASK_LIST_RERANKING
    + TASK_LIST_RETRIEVAL
    + TASK_LIST_STS
)


def run_worker(sh_run_path: Path, model_name: str, task: str) -> None:
    subprocess.run(["bash", str(sh_run_path), model_name, task])


def main():
    # .sh file that takes model and task name as arguments
    sh_run_path = Path(argv[1])

    # model name to be passed to the worker
    model_name = argv[2]

    tasks = argv[3]

    if tasks == "all":
        for task in TASK_LIST:
            run_worker(sh_run_path, model_name, task)
    elif tasks == "classification":
        for task in TASK_LIST_CLASSIFICATION:
            run_worker(sh_run_path, model_name, task)
    elif tasks == "clustering":
        for task in TASK_LIST_CLUSTERING:
            run_worker(sh_run_path, model_name, task)
    elif tasks == "pair_classification":
        for task in TASK_LIST_PAIR_CLASSIFICATION:
            run_worker(sh_run_path, model_name, task)
    elif tasks == "reranking":
        for task in TASK_LIST_RERANKING:
            run_worker(sh_run_path, model_name, task)
    elif tasks == "retrieval":
        for task in TASK_LIST_RETRIEVAL:
            run_worker(sh_run_path, model_name, task)
    elif tasks == "sts":
        for task in TASK_LIST_STS:
            run_worker(sh_run_path, model_name, task)
    else:
        raise ValueError(f"Unknown task {tasks}")


if __name__ == "__main__":
    main()
