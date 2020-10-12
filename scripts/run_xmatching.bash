# The name of experiment
name=$2

# Create dirs and make backup
output=snap/xmatching/$name
mkdir -p $output/src/
cp -r xmatching $output/src/
cp $0 $output/run.bash

# Pre-training
CUDA_VISIBLE_DEVICES=$1 unbuffer python xmatching/main.py \
    --train-imgs mscoco_train,mscoco_nominival --valid-imgs mscoco_minival \
    --train-langs mscoco --valid-langs mscoco \
    --max-len 20 --dim 64 \
    --visn resnext101_32x8d --lang bert --lang-layers 4,3,2,1 \
    --lang-pretrained --visn-pretrained \
    --num-workers 8 --batchSize 256 --optim adam --lr 1e-3 --epochs 20 \
    --nodes 1 --nr 0 \
    --output $output ${@:3} | tee $output/log.log

