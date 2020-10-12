DATA_DIR=data/wiki-cased-untokenized/
TOKENIZER=roberta-base
python vokenization/tokenization/tokenize_dataset.py $DATA_DIR en.valid.raw $TOKENIZER
python vokenization/tokenization/tokenize_dataset.py $DATA_DIR en.test.raw $TOKENIZER
python vokenization/tokenization/tokenize_dataset.py $DATA_DIR en.train.raw $TOKENIZER
