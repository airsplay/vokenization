# Vokenization

PyTorch code for the EMNLP 2020 paper "Vokenization: Improving Language Understanding with Contextualized, 
Visual-Grounded Supervision".

```shell script
pip install -r requirements.txt
```

## xmatching: Contextualized Cross-Modal Matching
In this module (corresponding to Sec 3.2 of the paper), 
we mainly want to learn a matching model, 
which "Contextually" measure the alignment between words and images. 
The terminology "Contextual" emphasize the nature that the sentences (the context) are also taken into consideration.

### Download Data
1. Download MS COCO images:
    ```shell script
    # MS COCO
    mkdir -p data/mscoco
    wget http://images.cocodataset.org/zips/train2014.zip -P data/mscoco
    wget http://images.cocodataset.org/zips/val2014.zip -P data/mscoco
    unzip data/mscoco/train2014.zip -d data/mscoco/images/ && rm data/mscoco/train2014.zip
    unzip data/mscoco/val2014.zip -d data/mscoco/images/ && rm data/mscoco/val2014.zip
    ```

2. Download captions (split following the LXMERT project):
    ```shell script
    mkdir -p data/lxmert
    wget https://nlp.cs.unc.edu/data/lxmert_data/lxmert/mscoco_train.json -P data/lxmert/
    wget https://nlp.cs.unc.edu/data/lxmert_data/lxmert/mscoco_nominival.json -P data/lxmert/
    wget https://nlp.cs.unc.edu/data/lxmert_data/lxmert/vgnococo.json -P data/lxmert/
    wget https://nlp.cs.unc.edu/data/lxmert_data/lxmert/mscoco_minival.json -P data/lxmert/
    ```

### Training Models
The model has a wide range support of different visn/lang backbones. 
For visual backbones, the models in torchvision are mostly supported. You might need to handle the last FC layer, 
because it is written differently in different backbones.
For language backbones, we utilize the models in [pytorch.transformers](https://github.com/huggingface/transformers).
The list of supporting models is provided in [CoX/model.py]() and could be extended following the guidelines in
[pytorch.transformers](https://github.com/huggingface/transformers).

Running Commands:
```bash
# Run the cross-modal matching model with single-machine multi-processing distributed training
# Speed: 20 min ~ 30 min / 1 Epoch, 20 Epochs by default.
bash scripts/run_xmatching.bash 0,1 bert_resnext

# Or you could fine-tune the model 
# bash finetune_cox.bash
```

## vokenization: Contextualized Retrieval
This step is a bridge between the cross-modality (words-and-image) matching models (CoR) and 
language-only pre-training models (CoL).
The final goal is vokenization, which converts the language tokens to related images 
(we called them **vokens**).
We mainly provide preprocessing tools (i.e., feature extraction, tokenization, and vokenization) and
evaluation tools of previous CoR models here.
Oveerall, we have four different utils in this stage, the pipeline is:
```
Extracting Image Features-----> Benchmakring the Matching Models (Optional) --> Vokenization
Tokenization --------------/
```

### Evaluating CoX Model and Vokenization
The evaluation includes two different 
```bash
bash scripts/cox_benchmarking.bash 0 bert_resnext
```


### Pure Language Data Preprocessing
We provide scripts to get the datasets "wiki103", "wiki", and "Book Corpus".
We would note them as "XX-cased" or "XX-uncased" where the suffix "cased" / "uncased" only indicates
the property of the raw text.
#### Wiki103
```bash
bash data/wiki103/get_data_cased.sh
```

#### English Wikipedia
The script to download and process wiki data are copied from [XLM](https://github.com/facebookresearch/XLM)
```bash
bash data/wiki/get_data_cased.bash en
```

Note: For RoBERTa, it requires an untokenized version of wiki, 
so please use the following command:
```bash
bash data/wiki/get_data_cased_untokenized.bash en
```

### Extracting Image Features
The image preprocessing first extracts the image features to build the keys for retrieval.

We first build a list of image indexes with `CoR/create_image_ids.py`. 
It is used to unify the image ids in different experiments thus the feature array stored in hdf5 could be universally indexed.
The image ids are saved under a shared path called `LOCAL_DIR` defined in `CoR/common.py`.
The path is used to save meta info for the retrieval and we will make sure all the experiments agree with this meta info,
so that we would not get different indexing in different retrieval experiments.
The image ids are saved under `{LOCAL_DIR}/images` with format `{IMAGE_SET}_ids.txt`.

> Note: The ids created by `create_image_ids.py` are only the order of the images.
> The actual images in the dictionary are provided by `extract_keys.bash`, thus is corresponding to the 
> `_paths.txt`, because the `extract_keys` will filter all broken images and non-existing images.

Commands:
```bash
# Step 1, Build image orders.
python col/cor/build_image_orders.py  
```
Extract image features regarding the list built above, using code `CoR/extract_vision_keys.py`. 
The code will first read the image ids saved in `{LOCAL_DIR}/images/{IMAGE_SET}_ids.txt` and locate the images.
The features will be saved under `{LOAD}/keys/coco_minival.hdf5`.
Commands:
```bash
# Step 2, Extract features. 
# bash extract_keys.bash $GPU_ID $MODEL_NAME 
bash extract_keys.bash 0 bert_resnext 
```

### Tokenization
The language preprocessing next tokenize the language corpus.
It split the paragraphs into sentences and tokenize each sentences to CoL model specific tokens (w.r.t. RoBERTa tokenization).
It would locally save three files: `{dset_path}.{tokenizer_name}`, `{dset_path}.{tokenizer_name}.sent`, 
and `{dset_path}.{tokenizer_name}.dcmt`.
`{dset_path}.{tokenizer_name}` saves the tokens and each line in this file is the tokens of a sentence.
`{dset_path}.{tokenizer_name}.sent` saves the starting token index of each sentence, it has `S+1` lines where `S` is the number of sentences.
Each sentence has tokens from `token[sent[i]: sent[i+1]]`.
`{dset_path}.{tokenizer_name}.dcmt` saves the starting sentence index of each document, it has `D+1` lines where `D` is the number of documents.
Each document has sentences from `sent[sent[i]: sent[i+1]]`.

Lastly, `to_hdf5.py` will save the processed tokens to hdf5 files `{dset_path}.{tokenizer_name}.hdf5`, locally.
The sentences could be directly fetched from this hdf5 files with the index in `.sent`.

```bash
# Tokenization
python CoR/preprocess/tokenize_wiki103.py
python CoR/preprocess/tokenize_wiki_all.py
python CoR/preprocess/tokenize_bc.py

# Save tokens to hdf5 files.
python to_hdf5.py       # May need to comment out some datasets.
```

New tokenization:
```bash
cd CoR/preprocess_bert/
bash tokenize_wiki.bash 
```


### Benchmarking Matching Models
> Before evaluating, please make sure that `extracting_image_features` and `tokenization` are completed.

We benchmark the performance of CoX models from large scale

### The Vokenization Process
After all these steps, we could start to vokenize the language corpus.
It would load the tokens saved in `{corpus}.{tokenizer_name}.hdf5` 
and uses the sentence info in `{corpus}.{tokenzier_name}.hdf5`.

The code is highly optimized and could be continued by just rerunning it.
The vokens will be saved in `{load}/vokens/wiki.train.raw.vg_nococo.hdf5` by default.
The file `{load}/vokens/wiki.train.raw.vg_nococo` contains the line-split vokens.
If you want to change the voken's output dir, please specify the option `--output`.

```bash
python CoR/vokenize_corpus.py \
--load /ssd-playpen/home/hTan/CoL/CoX/snap/pretrain/robertal4_finetune_from_pretrained \
--corpus=/ssd-playpen/data/wiki103/wiki.train.raw \
--image-sets=vg_nococo \
--tokenizer-name=roberta-base   # The tokenizer-name must be provided!
```

> Note: `--tokenizer-name` must be provided in the script.
> Moreover, please make sure that the tokens are extracted from [tokenization](#tokenization)

For the wiki and wiki103, use the following script:
#### Wiki103
First modifying the `LOAD=/ssd-playpen/home/hTan/CoL/CoX/snap/pretrain/cc_hinge05_dim64_resxt101_robertal4` in `vokenize_wiki103.bash`
to the saved CoX model.
```bash
bash vokenize_wiki103.bash 0
```
#### Wiki (all)
```bash
bash mpvokenize_wiki.bash 0,1,2,3
```



## vlm: Visually-Supervised Language Model

### RoBERTa Pre-Training

### GLUE Fine-Tuning
#### Evaluate the last checkpoint
By running
```bash
bash run_glue.bash 0 3
```
Run GLUE on GPU 0 for 3 epochs, then using
```bash
python CoL/show_glue_results.py
```
to check the GLUE results.

#### Evaluate multiple checkpoints

By running
```bash
bash run_glue_epochs.bash 0,1,2,3 --snaps 7                            
```
It will assess 7 snaps using all 0,1,2,3 GPUs. 
`snaps=-1` will assess all checkpoints.
Then using 
```bash
python CoL/show_glue_results_epochs.py 
```
to check all glue_results.
