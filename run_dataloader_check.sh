

sbatch --job-name=training \
      --account=ehz@v100 \
      --no-requeue \
      --cpus-per-task=8 \
      --hint=nomultithread \
      --gres=gpu:1 \
      --partition=gpu_p2 \
      --time=16:00:00 \
      --output=jobinfo_gpt_training/testlib%j.out \
      --error=jobinfo_gpt_training/testlib%j.err \
      --wrap="module purge; module load pytorch-gpu/py3/2.1.1;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python scripts/check_dataloading.py --lr 0.0001 --train_normalize --batch_size 6 \
         --gradient_accumulation_step 256 \
          --model_name Alibaba-NLP/gte-large-en-v1.5 \
           --teachers /gpfsscratch/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/jamesgpt1/sf_model_e5  /gpfsscratch/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Alibaba-NLP/gte-Qwen1.5-7B-instruct /gpfsscratch/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Alibaba-NLP/gte-Qwen2-7B-instruct/  /gpfsscratch/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/WhereIsAI/UAE-Large-V1        /gpfsscratch/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Alibaba-NLP/gte-Qwen2-1.5B-instruct /gpfsscratch/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/sentence-transformers/gtr-t5-xl /gpfsscratch/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Salesforce/SFR-Embedding-2_R /gpfsscratch/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/dunzhang/stella_en_400M_v5 /gpfsscratch/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Snowflake/snowflake-arctic-embed-m
        "
