for model in  "Snowflake/snowflake-arctic-embed-l"; do
sbatch --job-name=snowflake \
      --account=ehz@a100 \
      --gres=gpu:4 \
      --partition=gpu_p5 \
      --no-requeue \
      --cpus-per-task=32 \
      --hint=nomultithread \
      --time=16:00:00 \
      -C a100 \
      --output=jobinfo_snow_training/testlib%j.out \
      --error=jobinfo_snow_training/testlib%j.err \
      --wrap="module purge;  module load arch/a100; module load pytorch-gpu/py3/2.1.1;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python ../scripts/train_snowflakes.py --lr 0.00005 --batch_size 64 \
         --gradient_accumulation_step 8 \
         --experiment_name textdistill-snowflake-l-nll-gist \
          --model_name $model \
           --teachers /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings_gist/jamesgpt1/sf_model_e5  /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings_gist/WhereIsAI/UAE-Large-V1          /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings_gist/Salesforce/SFR-Embedding-2_R /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings_gist/dunzhang/stella_en_400M_v5
        "
  done
