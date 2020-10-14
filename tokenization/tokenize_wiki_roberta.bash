DATA_DIR=data/wiki-cased-untokenized/
TOKENIZER=roberta-base
python tokenization/tokenize_dataset.py $DATA_DIR en.valid.raw $TOKENIZER
python tokenization/tokenize_dataset.py $DATA_DIR en.test.raw $TOKENIZER
python tokenization/tokenize_dataset.py $DATA_DIR en.train.raw $TOKENIZER
