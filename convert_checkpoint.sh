

model_path=$1
output_dir=$2
backbone=$3

filename=$(basename -- "$model_path")
filename="${filename%.*}"


chmod +x run_mteb_distributed.sh
python scripts/extract_huggingface_model_from_pl_ckpt.py "$model_path" "$output_dir"/$backbone "$backbone"
python scripts/eval_english_mteb_slurm_distributed.py "$WORK"/EMIRR/textemb_distil/run_mteb_distributed.sh "$output_dir"/"$backbone"/"$filename" classification
python scripts/eval_english_mteb_slurm_distributed.py "$WORK"/EMIRR/textemb_distil/run_mteb_distributed.sh "$output_dir"/"$backbone"/"$filename" retrieval
python scripts/eval_english_mteb_slurm_distributed.py "$WORK"/EMIRR/textemb_distil/run_mteb_distributed.sh "$output_dir"/"$backbone"/"$filename" pair_classification
python scripts/eval_english_mteb_slurm_distributed.py "$WORK"/EMIRR/textemb_distil/run_mteb_distributed.sh "$output_dir"/"$backbone"/"$filename" sts
python scripts/eval_english_mteb_slurm_distributed.py "$WORK"/EMIRR/textemb_distil/run_mteb_distributed.sh "$output_dir"/"$backbone"/"$filename" clustering
