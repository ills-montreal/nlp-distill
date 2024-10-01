

MODEL=$1
TASK=$2
sbatch --job-name=eval_mteb \
      --account=ehz@v100 \
      --no-requeue \
      --cpus-per-task=8 \
      --hint=nomultithread \
      --gres=gpu:1 \
      --partition=gpu_p2 \
      --time=2:00:00 \
      --output=jobinfo_gpt_mteb_eval/testlib%j.out \
      --error=jobinfo_gpt_mteb_eval/testlib%j.err \
      --wrap="module purge; module load pytorch-gpu/py3/2.1.1;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python scripts/eval_english_mteb_slurm_worker.py $MODEL $TASK
        "
