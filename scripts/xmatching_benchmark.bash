# Benchmarking the cross-modal matching model with
#     1. Retrieval scores.
#     2. Voken Diversity w.r.t words in specific language corpus.
# Please run this after image_key_retrivel and tokenization. 
#    i.e., step 1 and step2 in readme.md

MODEL=$2
MODELPATH=snap/xmatching/$MODEL
rm -rf $MODELPATH/analysis.log

# Retrieval scores
CUDA_VISIBLE_DEVICES=$1 unbuffer python vokenization/evaluate_retrieval.py \
    --load $MODELPATH \
    --image-sets coco_minival,cc_valid \
    | tee -a $MODELPATH/analysis.log

# Diversity
# Test diversity of vision-and-language (captioning) datasets
CUDA_VISIBLE_DEVICES=$1 unbuffer python vokenization/evaluate_diversity.py \
    --load $MODELPATH \
    --image-sets vg_nococo \
    --corpus coco_minival,cc_valid \
    | tee -a $MODELPATH/analysis.log

# Test diversity of pure-language corpus
CUDA_VISIBLE_DEVICES=$1 unbuffer python vokenization/evaluate_diversity.py \
    --load $MODELPATH \
    --image-sets vg_nococo \
    --corpus data/wiki103-cased/wiki.valid.raw \
    --maxsents 95000 \
    | tee -a $MODELPATH/analysis.log
