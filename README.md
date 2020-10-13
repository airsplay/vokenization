# Vokenization

PyTorch code for the EMNLP 2020 paper "Vokenization: Improving Language Understanding with Contextualized, 
Visual-Grounded Supervision".

```shell script
pip install -r requirements.txt
```

## Contextualized Cross-Modal Matching (xmatching)
In this module (corresponding to Sec 3.2 of the paper), 
we mainly want to learn a matching model, 
which "Contextually" measure the alignment between words and images. 
The terminology "Contextual" emphasize the nature that the sentences (the context) are also taken into consideration.

### Download Image and Captioning Data
1. Download MS COCO images:
    ```shell script
    # MS COCO (Train 13G, Valid 6G)
    mkdir -p data/mscoco
    wget http://images.cocodataset.org/zips/train2014.zip -P data/mscoco
    wget http://images.cocodataset.org/zips/val2014.zip -P data/mscoco
    unzip data/mscoco/train2014.zip -d data/mscoco/images/ && rm data/mscoco/train2014.zip
    unzip data/mscoco/val2014.zip -d data/mscoco/images/ && rm data/mscoco/val2014.zip
    ```
   If you already have COCO image on disk. Save them as 
    ```
    data
      |-- mscoco
            |-- images
                 |-- train2014
                         |-- COCO_train2014_000000000009.jpg
                         |-- COCO_train2014_000000000025.jpg
                         |-- ......
                 |-- val2014
                         |-- COCO_val2014_000000000042.jpg
                         |-- ......
    ```

2. Download captions (split following the LXMERT project):
    ```shell script
    mkdir -p data/lxmert
    wget https://nlp.cs.unc.edu/data/lxmert_data/lxmert/mscoco_train.json -P data/lxmert/
    wget https://nlp.cs.unc.edu/data/lxmert_data/lxmert/mscoco_nominival.json -P data/lxmert/
    wget https://nlp.cs.unc.edu/data/lxmert_data/lxmert/vgnococo.json -P data/lxmert/
    wget https://nlp.cs.unc.edu/data/lxmert_data/lxmert/mscoco_minival.json -P data/lxmert/
    ```

### Training The Cross-Modal Matching Model
The model has a wide range support of different visn/lang backbones. 
For visual backbones, the models in torchvision are mostly supported. You might need to handle the last FC layer, 
because it is written differently in different backbones.
For language backbones, we utilize the models in [pytorch.transformers](https://github.com/huggingface/transformers).
The list of supporting models is provided in [CoX/model.py]() and could be extended following the guidelines in
[pytorch.transformers](https://github.com/huggingface/transformers).

Running Commands:
```bash
# Run the cross-modal matching model with single-machine multi-processing distributed training
# "0,1" indicates using the GPUs 0 and 1.
# "bert_resnext" is the name of this snapshot and would be saved at snap/xmatching/bert_resnext
# Speed: 20 min ~ 30 min / 1 Epoch, 20 Epochs by default.
bash scripts/run_xmatching.bash 0,1 bert_resnext
```

## Vokenization (vokenization)
The vokenization is a bridge between the cross-modality (words-and-image) matching models (xmatching) and 
visually-supervised lagnauge models (vlm).
The final goal is to convert the language tokens to related images 
(we called them **vokens**).
These **vokens** enable the visual supervision of the language model.
We mainly provide pr-eprocessing tools (i.e., feature extraction, tokenization, and vokenization) and
evaluation tools of previous cross-modal matching models here.
Here is a diagram of these processes and we next discuss them one-by-one:
```
Extracting Image Features-----> Benchmakring the Matching Models (Optional) --> Vokenization
Downloading Language Data --> Tokenization -->-->--/
```

### Downloading and Pre-processing Pure-Language Data 
We provide scripts to get the datasets "wiki103" and "wiki".
We would note them as "XX-cased" or "XX-uncased" where the suffix "cased" / "uncased" only indicates
the property of the raw text.
1. **Wiki103**.
    ```shell script
    bash data/wiki103/get_data_cased.sh
    ```
2. **English Wikipedia**. 
The script to download and process wiki data are modified from [XLM](https://github.com/facebookresearch/XLM).
It will download a 17G file. 
The speed depends on the networking and it usually takes several hours to filter the data.
    ```shell script
    bash data/wiki/get_data_cased.bash en
    ```
    Note: For *RoBERTa*, it requires an untokenized version of wiki (o.w. the results would be much lower), 
    so please use the following command:
    ```shell script
    bash data/wiki/get_data_cased_untokenized.bash en
    ```
   
### Tokenization of Language Data
We next tokenize the language corpus.
It would locally save three files: 
"$dataset_name.$tokenizer_name", 
"$dataset_name.$tokenizer_name.hdf5",
and "$dataset_name.$tokenizer_name.line".
Taking the wiki103 dataset and BERT tokenizer as an example, 
we convert the training file into
```
data 
 |-- wiki103-cased 
        |-- wiki.train.raw.bert-base-uncased
        |-- wiki.train.raw.bert-base-uncased.hdf5
        |-- wiki.train.raw.bert-base-uncased.line
```
The txt file `wiki.train.raw.bert-base-uncased` saves the tokens and each line in this file is the tokens of a line 
in the original file,
The hdf5 file `wiki.train.raw.bert-base-uncased.hdf5` stores all the tokens continuously and use
`wiki.train.raw.bert-base-uncased.line` to index the starting token index of each line.
The ".line" file has `L+1` lines where `L` is the number of lines in the original files.
Each line has a range "line[i]" to "line[i+1]" in the hdf5 file.

Commands:
1. Wiki103 (around 10 min)
    ```shell script
    bash vokenization/tokenization/tokenize_wiki103_bert.bash 
    ```
2. English Wikipedia ()
    ```shell script
    bash vokenization/tokenization/tokenize_wiki103_bert.bash 
    ```

### Extracting Image Features
The image pre-processing extracts the image features to build the keys in the vokenization retrieval process.

#### Download the Visual Genome (VG) images
```shell script
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -P data/vg/
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -P data/vg/
unzip data/vg/images.zip -d data/vg/images && rm data/vg/images.zip
unzip data/vg/images2.zip -d data/vg/images && rm data/vg/images2.zip
cd data/vg/images
mv VG_100K/* .
mv VG_100K_2/* .
rm -rf VG_100K VG_100K_2
cd ../../../
```
If you already have Visual Genome image on disk. Save them as 
```
data
|-- vg
    |-- images
         |-- 1000.jpg
         |-- 1001.jpg
         |-- ......
```
    
#### Build Universal Image Ids
We first build a list of image indexes with `vokenization/create_image_ids.py`. 
It is used to unify the image ids in different experiments 
thus the feature array stored in hdf5 could be universally indexed.
The image ids are saved under a shared path called `LOCAL_DIR` (default to `data/vokenization`)
 defined in `vokenization/common.py`.
The path is used to save meta info for the retrieval and we will make sure all the experiments agree with this meta info,
so that we would not get different indexing in different retrieval experiments.
The image ids are saved under `{LOCAL_DIR}/images` with format `{IMAGE_SET}_ids.txt`.

> Note: The ids created by `create_image_ids.py` are only the order of the images.
> The actual images in the dictionary are provided by `extract_keys.bash`, thus is corresponding to the 
> `_paths.txt`, because the `extract_keys` will filter all broken images and non-existing images.

Commands:
```bash
# Step 1, Build image orders.
python vokenization/build_image_orders.py  
```

#### Extracting Image Features

Extract image features regarding the list built above, using code `vokenization/extract_vision_keys.py`. 
The code will first read the image ids saved in `data/vokenization/images/{IMAGE_SET}_ids.txt` and locate the images.
The features will be saved under `snap/xmatching/bert_resnext/keys/{IMAGE_SET}.hdf5`.

Commands:
```bash
# Step 2, Extract features. 
# bash scripts/extract_keys.bash $GPU_ID $MODEL_NAME 
bash scripts/extract_keys.bash 0 bert_resnext 
```


### Benchmarking Cross-Modal Matching Models (Optional)
> Before evaluating, please make sure that `extracting_image_features` and `tokenization` are completed.

We benchmark the performance of cross-modal matching models from large scale.
The evaluation includes two different metrics: diversity and the retrieval performance.
```bash
bash scripts/xmatching_benchmark.bash 0 bert_resnext
```

### The Vokenization Process
After all these steps, we could start to vokenize the language corpus.
It would load the tokens saved in `dataset_name.tokenizer_name.hdf5` 
and uses the line-split information in `dataset_name.tokenzier_name.line`.

The code is optimized and could be continued by just rerunning it.
The vokens will be saved in `snap/xmatching/bert_resnext/vokens/wiki.train.raw.vg_nococo.hdf5` by default.
The file `snap/xmatching/bert_resnext/vokens/wiki.train.raw.vg_nococo.ids` contains the universal image ids 
for each voken, 
e.g., the image id `vg_nococo/8` corresponds to 8-th feature
saved in `snap/xmatching/bert_resnext/keys/vg_nococo.hdf5`.


> Note: `--tokenizer-name` must be provided in the script.
> Moreover, please make sure that the tokens are extracted from [tokenization](#tokenization)


Commands
1. Wiki103
    ```shell script
    # bash scripts/mpvokenize_wiki103.bash $USE_GPUS $SNAP_NAME
    bash scripts/mpvokenize_wiki103.bash 0,1,2,3 bert_resnext
    ```
2. English Wikipedia
    ```shell script
    # bash scripts/mpvokenize_wiki.bash $USE_GPUS $SNAP_NAME
    bash scripts/mpvokenize_wiki.bash 0,1,2,3 bert_resnext
    ```



## Visually-Supervised Language Model (vlm)

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
