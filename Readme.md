


# NLP Multi Teacher Distillation

## Usage of the repository

One should use `train_snowflakes` to reproduce our experiments.

```Example of usage
python scripts/train_snowflakes.py --lr 0.00005 --batch_size 256 \
      --gradient_accumulation_step 8 \
      --experiment_name textdistill-snowflake-nll-gist \
      --model_name Snowflake/snowflake-arctic-embed-xs \
      --teachers nlp_embeddings/embeddings_gist/jamesgpt1/sf_model_e5  
           nlp_embeddings/embeddings_gist/WhereIsAI/UAE-Large-V1          
           nlp_embeddings/embeddings_gist/Salesforce/SFR-Embedding-2_R 
           nlp_embeddings/embeddings_gist/dunzhang/stella_en_400M_v5
```


## Organization of the repository

The repository is organized as follows:
```
.
├── jupyters # All the figures in the paper. Fix Export variables to use
│   ├── fig_classification_against_single.ipynb
│   ├── fig_classification_full_tables.ipynb
│   ├── fig_classification_rank_matrices.ipynb
│   ├── fig_Classification tasks Pareto Frontier 2.ipynb
│   ├── fig_Clustering 2 tasks Pareto Frontier.ipynb
│   ├── fig_clustering_full_tables.ipynb
│   ├── fig_Clustering tasks Pareto Frontier.ipynb
│   ├── fig_global_ranking_classif_clustering.ipynb
│   ├── fig_global_ranking_classif_clustering_philip_csv.ipynb
│   ├── fig_sts_full_tables.ipynb
│   ├── fig_STS tasks Pareto Frontier.ipynb
│   ├── global_table.csv
│   ├── img_single_teacher.ipynb
│   ├── Impact of distillation on performances.ipynb
│   ├── __pycache__
│   ├── table_initial_student_and_teacher_perfs.ipynb
│   ├── Training dataset statistics.ipynb
│   └── visu_utils.py
├── non_sync # Dumped results, a bit heavy
│   ├── baselines_mteb
│   └── mteb_benchmarking
├── pyproject.toml
├── Readme.md
├── scripts
│   ├── cache_models.py # to cache models on nodes without internet
│   ├── cache_mteb_ds.py
│   ├── check_dataloading.py
│   ├── eval_english_mteb.py
│   ├── eval_english_mteb_slurm_distributed.py
│   ├── eval_english_mteb_slurm_worker.py
│   ├── extract_huggingface_model_from_pl_ckpt.py
│   ├── extract_huggingface_model_from_pl_ckpt_teacher_reconstruction.py
│   ├── generate_teacher_embeddings.py
│   ├── train_embedder_from_scratch.py # initial training script
│   ├── train_snowflakes.py # better version for training snowflakes
│   └── utils # utils for training and visualization, pytorchlightning models
└── training_dataset # scripts to create the hf dataset
├── gist_dataset_export_hf.py
└── load_merge_and_export_hf_dataset.py
```