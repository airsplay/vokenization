GPUS=$1
# The name of experiment
NAME=$2

# Create dirs and make backup
output=snap/bert/$NAME
mkdir -p $output/src
cp -r vlm/*.py $output/src/
cp $0 $output/run.bash
cp run_glue_epochs.bash $output/run_glue_epochs.bash
cp run_glue_at_epoch.bash $output/run_glue_at_epoch.bash 

export TRAIN_FILE=data/wiki-cased/en.train.raw
export TEST_FILE=data/wiki-cased/en.valid.raw

# Pre-training
CUDA_VISIBLE_DEVICES=$GPUS unbuffer python vlm/run_lm_distributed.py \
    --output_dir=$output \
	--overwrite_output_dir \
	--config_name=vlm/configs/bert-12L-768H.json \
	--tokenizer_name=bert-base-uncased \
    --model_type=bert \
	--block_size=126 \
	--per_gpu_train_batch_size=64 \
    --per_gpu_eval_batch_size=64 \
	--gradient_accumulation_steps=1 \
    --max_steps 220000 \
	--learning_rate=2e-4 \
	--weight_decay=0.01 \
	--warmup_steps=5000 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --col_data \
    --split_sent \
    --mlm ${@:3} | tee $output/log.log

    #--fp16 \
	#--fp16_opt_level O2 \

#--shuffle \
# Wait for clearing the GPU cache
sleep 30
bash scripts/run_glue_epochs.bash $GPUS $output --snaps -1
