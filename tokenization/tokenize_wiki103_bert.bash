DATA_DIR=data/wiki103-cased
TOKENIZER=bert-base-uncased
python tokenization/tokenize_dataset.py $DATA_DIR wiki.valid.raw $TOKENIZER
python tokenization/tokenize_dataset.py $DATA_DIR wiki.test.raw $TOKENIZER
python tokenization/tokenize_dataset.py $DATA_DIR wiki.train.raw $TOKENIZER
