export GLUE_DIR=data/glue/
EPOCHS=$2
MODEL=$3
CKPT=$4

for TASK_NAME in WNLI RTE MRPC STS-B CoLA SST-2 QNLI QQP MNLI
do
    CUDA_VISIBLE_DEVICES=$1 python vlm/run_glue.py \
        --model_type bert \
        --tokenizer_name=bert-base-uncased \
        --model_name_or_path $MODEL/$CKPT \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --do_lower_case \
        --data_dir $GLUE_DIR/$TASK_NAME \
        --save_steps -1 \
        --max_seq_length 126 \
        --per_gpu_eval_batch_size=32   \
        --per_gpu_train_batch_size=32   \
        --learning_rate 1e-4 \
        --warmup_steps 0.1 \
        --num_train_epochs $EPOCHS.0 \
        --output_dir $MODEL/glueepoch_$CKPT/$TASK_NAME
done

        #--overwrite_output_dir \
