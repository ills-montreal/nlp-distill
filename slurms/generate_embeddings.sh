

# BIG MODELS
# "Salesforce/SFR-Embedding-Mistral"  "jspringer/echo-mistral-7b-instruct-lasttoken"  "Alibaba-NLP/gte-Qwen1.5-7B-instruct"
#  "izhx/udever-bloom-560m"
# "google/gemma-7b-it" "jspringer/echo-mistral-7b-instruct-lasttoken" "sentence-transformers/gtr-t5-xl" "Salesforce/SFR-Embedding-Mistral"
# "Alibaba-NLP/gte-Qwen1.5-7B-instruct"
#"Salesforce/SFR-Embedding-2_R"

for model in "Alibaba-NLP/gte-Qwen2-7B-instruct"; do
    sbatch --job-name=emb2 \
      --account=ehz@a100 \
      --gres=gpu:1 \
      --partition=gpu_p5 \
      --no-requeue \
      --cpus-per-task=16 \
      --hint=nomultithread \
      --time=16:00:00 \
      -C a100 \
      --output=jobinfo_emb/big%j.out \
      --error=jobinfo_emb/big%j.err \
      --wrap="module purge; module load cpuarch/amd ; module load pytorch-gpu/py3/2.1.1;
        export HF_DATASETS_OFFLINE=1;
        export TRANSFORMERS_OFFLINE=1;
        python ../scripts/generate_teacher_embeddings.py \
        --model ${model} \
        --dataset Icannos/distillation_training_gist_medi_mteb \
        --batch_size 8 \
        --output_dir $SCRATCH/EMIR/nlp_embeddings/embeddings_gist \
        --start 0 \
        --flash_attn"
done

# Small-ish models
#for model in "WhereIsAI/UAE-Large-V1" "infgrad/stella-base-en-v2" "sentence-transformers/LaBSE" "jamesgpt1/sf_model_e5" "sentence-transformers/all-MiniLM-L6-v2"; do
#for model in  "Alibaba-NLP/gte-Qwen2-1.5B-instruct" "dunzhang/stella_en_1.5B_v5"; do # "Snowflake/snowflake-arctic-embed-s" "Snowflake/snowflake-arctic-embed-m"; do
#    sbatch --job-name=emb2 \
#      --account=ehz@v100 \
#      --gres=gpu:1 \
#      --partition=gpu_p2 \
#      --no-requeue \
#      --cpus-per-task=10 \
#      --hint=nomultithread \
#      --time=16:00:00 \
#      --output=jobinfo_emb/small%j.out \
#      --error=jobinfo_emb/small%j.err \
#      --wrap="module purge; module load pytorch-gpu/py3/2.1.1;
#        export HF_DATASETS_OFFLINE=1;
#        export TRANSFORMERS_OFFLINE=1;
#        python ../scripts/generate_teacher_embeddings.py \
#        --model ${model} \
#        --dataset Icannos/distillation_training_gist_medi_mteb \
#        --batch_size 32 \
#        --output_dir $SCRATCH/EMIR/nlp_embeddings/embeddings_gist \
#        --start 0 "
#done


#for model in "dunzhang/stella_en_400M_v5"; do # "Alibaba-NLP/gte-Qwen2-1.5B-instruct" "dunzhang/stella_en_1.5B_v5" "Snowflake/snowflake-arctic-embed-s" "Snowflake/snowflake-arctic-embed-m"; do
#    sbatch --job-name=emb2 \
#      --account=ehz@v100 \
#      --gres=gpu:1 \
#      --partition=gpu_p2 \
#      --no-requeue \
#      --cpus-per-task=10 \
#      --hint=nomultithread \
#      --time=16:00:00 \
#      --output=jobinfo_emb/small%j.out \
#      --error=jobinfo_emb/small%j.err \
#      --wrap="module purge; module load pytorch-gpu/py3/2.1.1;
#        export HF_DATASETS_OFFLINE=1;
#        export TRANSFORMERS_OFFLINE=1;
#        python ../scripts/generate_teacher_embeddings.py \
#        --model ${model} \
#        --dataset Icannos/distillation_training_gist_medi_mteb \
#        --batch_size 32 \
#        --output_dir $SCRATCH/EMIR/nlp_embeddings/embeddings_gist \
#        --start 0 \
#        --no_float16 "
#done


# for model in "dunzhang/stella_en_400M_v5" "Alibaba-NLP/gte-Qwen2-1.5B-instruct" "dunzhang/stella_en_1.5B_v5" "Snowflake/snowflake-arctic-embed-s" "Snowflake/snowflake-arctic-embed-m"; do
# for model in "Alibaba-NLP/gte-Qwen2-1.5B-instruct"; do
#    sbatch --job-name=emb2 \
#      --account=ehz@a100 \
#      --gres=gpu:1 \
#      --partition=gpu_p5 \
#      --no-requeue \
#      --cpus-per-task=8 \
#      --hint=nomultithread \
#      --time=16:00:00 \
#      -C a100 \
#      --output=jobinfo_emb/small%j.out \
#      --error=jobinfo_emb/small%j.err \
#      --wrap="module purge;  module load cpuarch/amd; module load pytorch-gpu/py3/2.1.1;
#        export HF_DATASETS_OFFLINE=1;
#        export TRANSFORMERS_OFFLINE=1;
#        python ../scripts/generate_teacher_embeddings.py \
#        --model ${model} \
#        --dataset Icannos/distillation_training_gist_medi_mteb \
#        --batch_size 64 \
#        --output_dir $SCRATCH/EMIR/nlp_embeddings/embeddings \
#        --start 0 \
#         --flash_attn"
#done
