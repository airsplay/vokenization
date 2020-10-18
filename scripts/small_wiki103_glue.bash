# The name of experiment
GPUS=$1
NAME=$2

# Create dirs and make backup
output=snap/bert/$NAME
mkdir -p $output/src
cp -r vlm/*.py $output/src/
cp $0 $output/run.bash

export TRAIN_FILE=data/wiki103-cased/wiki.train.raw
export TEST_FILE=data/wiki103-cased/wiki.valid.raw

# Pre-training
CUDA_VISIBLE_DEVICES=$GPUS unbuffer python vlm/run_lm_distributed.py \
    --output_dir=$output \
	--overwrite_output_dir \
	--config_name=vlm/configs/bert-6L-512H.json \
	--tokenizer_name=bert-base-uncased \
    --model_type=bert \
	--block_size=126 \
	--per_gpu_train_batch_size=64 \
    --per_gpu_eval_batch_size=64 \
	--gradient_accumulation_steps=1 \
	--num_train_epochs=44 \
	--learning_rate=2e-4 \
	--weight_decay=0.01 \
	--warmup_steps=10000 \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --col_data \
    --split_sent \
    --shuffle \
    --mlm ${@:3} | tee $output/log.log

    #--fp16 \
	#--fp16_opt_level O2 \


# Wait for clearing the GPU cache
sleep 30
bash scripts/run_glue_epochs.bash $GPUS $output --snaps 4
