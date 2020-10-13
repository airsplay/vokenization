GPU=$1

LOAD=snap/xmatching/$2
DATA_DIR=data/wiki-cased
TOKENIZER=bert-base-uncased

for DATA_NAME in en.valid.raw en.test.raw en.train.raw
do 
    CUDA_VISIBLE_DEVICES=$GPU python vokenization/vokenize_corpus_mp.py \
        --load $LOAD \
        --corpus=$DATA_DIR/$DATA_NAME \
        --tokenizer-name $TOKENIZER \
        --image-sets vg_nococo \
        --max-img-num 50000 
done

