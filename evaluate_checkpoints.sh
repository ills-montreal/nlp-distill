
model_path=$2
output_dir=$1
backbone=$3

  sbatch --job-name=convert \
      --account=ehz@v100 \
      --gres=gpu:1 \
      --no-requeue \
      --cpus-per-task=8 \
      --hint=nomultithread \
      --time=00:30:00 \
      --output=jobinfo_convert_and_run_eval/testlib%j.out \
      --error=jobinfo_convert_and_run_eval/testlib%j.err \
      --wrap="module purge;  module load pytorch-gpu/py3/2.1.1;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        chmod +x convert_checkpoint.sh;
        ./convert_checkpoint.sh $model_path $output_dir $backbone"

