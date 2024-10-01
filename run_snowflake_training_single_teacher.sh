
#
for model in "Snowflake/snowflake-arctic-embed-xs" "Snowflake/snowflake-arctic-embed-s"; do
sbatch --job-name=snowflake \
      --account=ehz@a100 \
      --gres=gpu:1 \
      --partition=gpu_p5 \
      --no-requeue \
      --cpus-per-task=16 \
      --hint=nomultithread \
      --time=16:00:00 \
      -C a100 \
      --output=jobinfo_snow_training/testlib%j.out \
      --error=jobinfo_snow_training/testlib%j.err \
      --wrap="module purge;  module load cpuarch/amd; module load pytorch-gpu/py3/2.1.1;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python scripts/train_snowflakes.py --lr 0.00005 --batch_size 256 \
         --gradient_accumulation_step 8 \
         --experiment_name textdistill-snowflake-nll_single_sfr-gist \
          --model_name $model \
           --teachers /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings_gist/Salesforce/SFR-Embedding-2_R
        "
done


# MSE

for model in "Snowflake/snowflake-arctic-embed-xs" "Snowflake/snowflake-arctic-embed-s" ; do
sbatch --job-name=snowflake \
      --account=ehz@a100 \
      --gres=gpu:1 \
      --partition=gpu_p5 \
      --no-requeue \
      --cpus-per-task=16 \
      --hint=nomultithread \
      --time=16:00:00 \
      -C a100 \
      --output=jobinfo_snow_training/testlib%j.out \
      --error=jobinfo_snow_training/testlib%j.err \
      --wrap="module purge;  module load cpuarch/amd; module load pytorch-gpu/py3/2.1.1;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python scripts/train_snowflakes.py --lr 0.00005 --batch_size 256 \
         --gradient_accumulation_step 8 \
         --experiment_name textdistill-snowflake-mse_single_sfr-gist \
         --MSE \
          --model_name $model \
           --teachers /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings_gist/Salesforce/SFR-Embedding-2_R
        "
done


