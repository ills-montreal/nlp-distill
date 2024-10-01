
sbatch --job-name=training \
      --account=ehz@a100 \
      --gres=gpu:1 \
      --partition=gpu_p5 \
      --no-requeue \
      --cpus-per-task=16 \
      --hint=nomultithread \
      --time=16:00:00 \
      -C a100 \
      --output=jobinfo_gpt_training/testlib%j.out \
      --error=jobinfo_gpt_training/testlib%j.err \
      --wrap="module purge;  module load cpuarch/amd; module load pytorch-gpu/py3/2.1.1;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python scripts/train_embedder_from_scratch.py --lr 0.0001 --batch_size 8 \
         --gradient_accumulation_step 128 \
         --experiment_name textdistill-mse \
         --experiment_id sisquidjkd \
          --model_name Alibaba-NLP/gte-large-en-v1.5 \
           --teachers /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/jamesgpt1/sf_model_e5  /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Alibaba-NLP/gte-Qwen2-7B-instruct/  /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/WhereIsAI/UAE-Large-V1          /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Salesforce/SFR-Embedding-2_R /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/dunzhang/stella_en_400M_v5
        "
