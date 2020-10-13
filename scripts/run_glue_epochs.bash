GPUS=$1
MODEL=$2
 
python vlm/run_glue_epochs.py --gpus $GPUS --load $MODEL \
    ${@:3}

