#
#sbatch --job-name=training \
#      --account=ehz@v100 \
#      --no-requeue \
#      --cpus-per-task=8 \
#      --hint=nomultithread \
#      --gres=gpu:1 \
#      --partition=gpu_p2 \
#      --time=16:00:00 \
#      --output=jobinfo_gpt_training/testlib%j.out \
#      --error=jobinfo_gpt_training/testlib%j.err \
#      --wrap="module purge; module load pytorch-gpu/py3/2.1.1;
#        export HF_DATASETS_OFFLINE=1;
#        export TRANSFORMERS_OFFLINE=1;
#        python scripts/train_embedder_from_scratch.py --lr 0.0001 --batch_size 8 --gradient_accumulation_step 16 --model_name Alibaba-NLP/gte-large-en-v1.5 --aligned_inputs --teachers  /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/jamesgpt1/sf_model_e5 /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Salesforce/SFR-Embedding-Mistral /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Alibaba-NLP/gte-Qwen1.5-7B-instruct /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/izhx/udever-bloom-560m /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/jspringer/echo-mistral-7b-instruct-lasttoken /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/infgrad/stella-base-en-v2 /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/WhereIsAI/UAE-Large-V1
#        "



#sbatch --job-name=training \
#      --account=ehz@v100 \
#      --no-requeue \
#      --cpus-per-task=8 \
#      --hint=nomultithread \
#      --gres=gpu:1 \
#      --partition=gpu_p2 \
#      --time=16:00:00 \
#      --output=jobinfo_gpt_training/testlib%j.out \
#      --error=jobinfo_gpt_training/testlib%j.err \
#      --wrap="module purge; module load pytorch-gpu/py3/2.1.1;
#        export HF_DATASETS_OFFLINE=1;
#        export TRANSFORMERS_OFFLINE=1;
#        python scripts/train_embedder_from_scratch.py --lr 0.0001 --train_normalize --batch_size 4 \
#         --gradient_accumulation_step 64 \
#          --model_name Alibaba-NLP/gte-large-en-v1.5 \
#           --teachers /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/jamesgpt1/sf_model_e5  /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Alibaba-NLP/gte-Qwen1.5-7B-instruct /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Alibaba-NLP/gte-Qwen2-7B-instruct/  /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/WhereIsAI/UAE-Large-V1        /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Alibaba-NLP/gte-Qwen2-1.5B-instruct /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/sentence-transformers/gtr-t5-xl /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Salesforce/SFR-Embedding-2_R /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/dunzhang/stella_en_400M_v5 /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Snowflake/snowflake-arctic-embed-m
#        "


#sbatch --job-name=training \
#      --account=ehz@v100 \
#      --no-requeue \
#      --cpus-per-task=8 \
#      --hint=nomultithread \
#      --gres=gpu:2 \
#      --partition=gpu_p2 \
#      --time=16:00:00 \
#      --output=jobinfo_gpt_training/testlib%j.out \
#      --error=jobinfo_gpt_training/testlib%j.err \
#      --wrap="module purge; module load pytorch-gpu/py3/2.1.1;
#        export HF_DATASETS_OFFLINE=1;
#        export TRANSFORMERS_OFFLINE=1;
#        python scripts/train_embedder_from_scratch.py --lr 0.0001 --train_normalize --batch_size 4 \
#         --gradient_accumulation_step 45 \
#          --model_name Alibaba-NLP/gte-large-en-v1.5 \
#           --teachers /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/jamesgpt1/sf_model_e5  /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Alibaba-NLP/gte-Qwen1.5-7B-instruct /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Alibaba-NLP/gte-Qwen2-7B-instruct/  /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/WhereIsAI/UAE-Large-V1        /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Alibaba-NLP/gte-Qwen2-1.5B-instruct /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/sentence-transformers/gtr-t5-xl /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Salesforce/SFR-Embedding-2_R /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/dunzhang/stella_en_400M_v5 /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Snowflake/snowflake-arctic-embed-m
#        "

# project="textdistill-5", id="ptg6gk4k")

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
         --experiment_name textdistill-5 \
         --experiment_id ptg6gk4k \
          --model_name Alibaba-NLP/gte-large-en-v1.5 \
           --teachers /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/jamesgpt1/sf_model_e5  /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Alibaba-NLP/gte-Qwen2-7B-instruct/  /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/WhereIsAI/UAE-Large-V1          /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Salesforce/SFR-Embedding-2_R /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/dunzhang/stella_en_400M_v5
        "









#sbatch --job-name=training \
#      --account=ehz@v100 \
#      --no-requeue \
#      --cpus-per-task=8 \
#      --hint=nomultithread \
#      --gres=gpu:1 \
#      --partition=gpu_p2 \
#      --time=16:00:00 \
#      --output=jobinfo_gpt_training/testlib%j.out \
#      --error=jobinfo_gpt_training/testlib%j.err \
#      --wrap="module purge; module load pytorch-gpu/py3/2.1.1;
#        export HF_DATASETS_OFFLINE=1;
#        export TRANSFORMERS_OFFLINE=1;
#        python scripts/train_embedder_from_scratch.py --lr 0.0001 --batch_size 6 --gradient_accumulation_step 32 --model_name Alibaba-NLP/gte-large-en-v1.5 --aligned_inputs --teachers  /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/jamesgpt1/sf_model_e5 /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Salesforce/SFR-Embedding-Mistral /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Alibaba-NLP/gte-Qwen1.5-7B-instruct /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/jspringer/echo-mistral-7b-instruct-lasttoken /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/WhereIsAI/UAE-Large-V1        /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Alibaba-NLP/gte-Qwen2-1.5B-instruct /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/sentence-transformers/gtr-t5-xl /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Salesforce/SFR-Embedding-2_R /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/dunzhang/stella_en_400M_v5 /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Snowflake/snowflake-arctic-embed-m
#        "

#sbatch --job-name=training \
#      --account=ehz@v100 \
#      --no-requeue \
#      --cpus-per-task=8 \
#      --hint=nomultithread \
#      --gres=gpu:1 \
#      --partition=gpu_p2 \
#      --time=16:00:00 \
#      --output=jobinfo_gpt_training/testlib%j.out \
#      --error=jobinfo_gpt_training/testlib%j.err \
#      --wrap="module purge; module load pytorch-gpu/py3/2.1.1;
#        export HF_DATASETS_OFFLINE=1;
#        export TRANSFORMERS_OFFLINE=1;
#        python scripts/train_embedder_from_scratch.py --lr 0.0001 --batch_size 8 --gradient_accumulation_step 64 --model_name Alibaba-NLP/gte-large-en-v1.5 --aligned_inputs --teachers  /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/jamesgpt1/sf_model_e5 /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Salesforce/SFR-Embedding-Mistral /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Alibaba-NLP/gte-Qwen1.5-7B-instruct /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/izhx/udever-bloom-560m /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/jspringer/echo-mistral-7b-instruct-lasttoken /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/infgrad/stella-base-en-v2 /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/WhereIsAI/UAE-Large-V1
#        "

# /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/jamesgpt1/sf_model_e5
# /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Salesforce/SFR-Embedding-Mistral
# /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/sentence-transformers/LaBSE
# /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/sentence-transformers/all-MiniLM-L6-v2
# /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Alibaba-NLP/gte-Qwen1.5-7B-instruct
# /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/izhx/udever-bloom-560m
# /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/jspringer/echo-mistral-7b-instruct-lasttoken
#  /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/infgrad/stella-base-en-v2
# /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/WhereIsAI/UAE-Large-V1
#

# python scripts/train_embedder_from_scratch.py --model_name Alibaba-NLP/gte-large-en-v1.5 --aligned_inputs --teachers  /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/jamesgpt1/sf_model_e5 /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Salesforce/SFR-Embedding-Mistral /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/Alibaba-NLP/gte-Qwen1.5-7B-instruct /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/izhx/udever-bloom-560m /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/jspringer/echo-mistral-7b-instruct-lasttoken /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/infgrad/stella-base-en-v2 /lustre/fsn1/projects/rech/ehz/uwf24rf/EMIR/nlp_embeddings/embeddings/WhereIsAI/UAE-Large-V1
