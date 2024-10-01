
chmod +x evaluate_checkpoints.sh

./evaluate_checkpoints.sh experiments_gist_nll  "checkpoints/textdistill-snowflake-nll-gist_Snowflake_snowflake-arctic-embed-xs-NLL/epoch=80-step=142000.ckpt"  Snowflake/snowflake-arctic-embed-xs
./evaluate_checkpoints.sh experiments_gist_nll  "checkpoints/textdistill-snowflake-nll-gist_Snowflake_snowflake-arctic-embed-s-NLL/epoch=54-step=96000.ckpt"  Snowflake/snowflake-arctic-embed-s
./evaluate_checkpoints.sh experiments_gist_nll  "checkpoints/textdistill-snowflake-nll-gist_Snowflake_snowflake-arctic-embed-m-NLL/epoch=38-step=68000.ckpt"  Snowflake/snowflake-arctic-embed-m
# ./evaluate_checkpoints.sh experiments_gist_nll  "checkpoints/textdistill-snowflake-nll-gist_Snowflake_snowflake-arctic-embed-l-NLL/epoch=6-step=12000-v2"  Snowflake/snowflake-arctic-embed-l

./evaluate_checkpoints.sh experiments_gist_mse  "checkpoints/textdistill-snowflake-mse-gist_Snowflake_snowflake-arctic-embed-xs-MSE/epoch=77-step=138000.ckpt"  Snowflake/snowflake-arctic-embed-xs
./evaluate_checkpoints.sh experiments_gist_mse  "checkpoints/textdistill-snowflake-mse-gist_Snowflake_snowflake-arctic-embed-s-MSE/epoch=48-step=86000.ckpt"  Snowflake/snowflake-arctic-embed-s
./evaluate_checkpoints.sh experiments_gist_mse  "checkpoints/textdistill-snowflake-mse-gist_Snowflake_snowflake-arctic-embed-m-MSE/epoch=35-step=62000.ckpt"  Snowflake/snowflake-arctic-embed-m
#./evaluate_checkpoints.sh experiments_gist_mse  "checkpoints/textdistill-snowflake-mse-gist_Snowflake_snowflake-arctic-embed-l-MSE/epoch=2-step=4000.ckpt"  Snowflake/snowflake-arctic-embed-l


./evaluate_checkpoints.sh experiments_gist_single_sfr_nll  "checkpoints/textdistill-snowflake-nll_single_sfr-gist_Snowflake_snowflake-arctic-embed-xs-NLL/epoch=45-step=80000.ckpt"  Snowflake/snowflake-arctic-embed-xs
./evaluate_checkpoints.sh experiments_gist_single_sfr_nll  "checkpoints/textdistill-snowflake-nll_single_sfr-gist_Snowflake_snowflake-arctic-embed-s-NLL/epoch=25-step=46000.ckpt"  Snowflake/snowflake-arctic-embed-s

./evaluate_checkpoints.sh experiments_gist_single_sfr_mse  "checkpoints/textdistill-snowflake-nll_single_sfr-gist_Snowflake_snowflake-arctic-embed-xs-MSE/epoch=45-step=80000.ckpt"  Snowflake/snowflake-arctic-embed-xs
./evaluate_checkpoints.sh experiments_gist_single_sfr_mse  "checkpoints/textdistill-snowflake-nll_single_sfr-gist_Snowflake_snowflake-arctic-embed-s-MSE/epoch=25-step=46000.ckpt"  Snowflake/snowflake-arctic-embed-s
