GPU=$1

LOAD=snap/xmatching/$2
WIKI_DIR=data/wiki103-cased
TOKENIZER=bert-base-uncased

for DATA_NAME in wiki.valid.raw wiki.test.raw wiki.train.raw
do 
    CUDA_VISIBLE_DEVICES=$GPU python vokenization/vokenize_corpus_mp.py \
        --load $LOAD \
        --corpus=$WIKI_DIR/$DATA_NAME \
        --tokenizer-name $TOKENIZER \
        --image-sets vg_nococo \
        --max-img-num 50000
done

