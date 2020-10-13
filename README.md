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

### Training the Cross-Modal Matching Model
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

### Downloading and Pre-Processing Pure-Language Data 
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
    bash vokenization/tokenization/tokenize_wiki_bert.bash 
    ```

### Extracting Image Features
The image pre-processing extracts the image features to build the keys in the vokenization retrieval process.

#### Download the Visual Genome (VG) images
Since MS COCO images are used in training the cross-modal matching model
as in [xmatching](#contextualized-cross-modal-matching-xmatching).
We will use the [Visual Genome](https://visualgenome.org/) images as 
candidate vokens for retrievel.
We here download the images first.
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
    # Note: mp is the abbreviation for "multi-processing"
    # bash scripts/mpvokenize_wiki103.bash $USE_GPUS $SNAP_NAME
    bash scripts/mpvokenize_wiki103.bash 0,1,2,3 bert_resnext
    ```
2. English Wikipedia
    ```shell script
    # bash scripts/mpvokenize_wiki.bash $USE_GPUS $SNAP_NAME
    bash scripts/mpvokenize_wiki.bash 0,1,2,3 bert_resnext
    ```



## Visually-Supervised Language Model (vlm)

### Pre-Training with VLM
As discussed in Sec. 2 of the paper,
we use visually-supervised language model to pre-train the model 
with visual supervision.
The visual supervision comes from the previous 
vokenization process.

#### Wiki103 
After the [vokenization process](#the-vokenization-process) of wiki103,
we could run the model with command:
```shell script
# bash scripts/small_vlm_wiki103_glue.bash $GPUs $SNAP_NAME
bash scripts/small_vlm_wiki103.bash 0,1,2,3 wiki103_bert_small
```
It will run a BERT-6Layers-512Hiddens model on [wiki103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)
dataset with the support of voken supervisions.
The snapshot will be saved to `snap/vlm/wiki103_bert_small`.
We recommend to run this Wiki103 experiment first since it will finish 
in a reasonable time (half a day).
The pure BERT pre-training option is also available [later](#bert-as-baselines)
for comparisons.

Note: defautly, the mixed-precision training is not used.
To support the mixed precision pre-training, 
please install the [nvidia/apex](https://github.com/NVIDIA/apex) library.
```shell script
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
After that, you could bring the option `--fp16` and `--fp16_opt_level O2` back in 
the script `scripts/small_vlm_wiki103_glue.bash`.
I recommend to use `--fp16_opt_level O2`.
Although the option O2 might be [unstable](https://github.com/NVIDIA/apex/issues/818#issuecomment-639012282),
it saves memory.
The per-gpu-batch-size is 32 with O1 but 64 with O2.

#### English Wikipedia
After the [vokenization process](#the-vokenization-process) of wiki103,
we could run the model with command:
```shell script
# bash scripts/small_vlm_wiki_glue.bash $GPUs $SNAP_NAME
bash scripts/base_vlm_wiki.bash 0,1,2,3 wiki_bert_base
```
It will run a BERT-12Layers-768Hiddens (same as BERT_BASE) model on the English Wikipedia
dataset with the support of voken supervisions.
The snapshot will be saved to `snap/vlm/wiki_bert_base`.

It takes around 3~5 days on 4 Titan V / GTX 2080
and around 5~7 days to finish in 4 Titan Pascal/T4 cards.
(This estimation is accurate since I inevitably run experiments on all these servers...).
Titan V / 2080 / T4 have native support of mixed precision training (triggered by `--fp16` option and need
installing [apex](https://github.com/NVIDIA/apex)).
The speed would be much faster.
Titan Pascal would also save some memory with the `--fp16` option.


### GLUE Evaluation
We defautly use the [GLUE](https://gluebenchmark.com/) benchmark
(e.g., [SST](https://nlp.stanford.edu/sentiment/index.html),
[MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398),
[QQP](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs),
[MNLI](https://cims.nyu.edu/~sbowman/multinli/),
[QNLI](https://rajpurkar.github.io/SQuAD-explorer/),)
 as downstreaming tasks.
Other tasks could be evaluated following the setup [here](https://github.com/huggingface/transformers/tree/28d183c90cbf91e94651cf4a655df91a52ea1033/examples)
by changing the option `--model_name_or_path` to the correct snapshot path `snap/bert/wiki103`.

#### Download GLUE dataset
This downloaindg scrip is copied from [huggingface transformers](https://github.com/huggingface/transformers/tree/master/examples/text-classification)
project.
Since the [transformers](https://github.com/huggingface/transformers) is still under dense
development, the change of APIs might affect the code. 
I have upgraded the code compaticability to transformers==3.3.
```shell script
wget https://raw.githubusercontent.com/huggingface/transformers/master/utils/download_glue_data.py
python download_glue_data.py --data_dir data/glue --tasks all
```

#### Finetuning on GLUE Tasks

1. Running GLUE evaluation for multiple snapshots:
    ```bash
    # bash scripts/run_glue_epochs.bash $GPUS #SNAP_PATH --snaps $NUM_OF_SNAPS                            
    bash scripts/run_glue_epochs.bash 0,1,2,3 snap/vlm/wiki103_bert_small --snaps 7                            
    ```
    It will assess 7 snaps using all 0,1,2,3 GPUs. 
    Setting `snaps=-1` will assess all checkpoints.
2. Running GLUE evaluation for one snapshots:
    ```shell script
    # bash scripts/run_glue_at_epoch.bash $GPUs $NUM_of_Finetuning_Epochs $SNAP_PATH $CHECKPOINT_NAME
    bash scripts/run_glue_at_epoch.bash 3 3 snap/vlm/wiki103_bert_small checkpoint-epoch0001
    ```

#### Showing the results
For all results saved under `snap/` (whatever the dir names),
running the folloing command will print out all the results.
```bash
python vlm/show_glue_results_epochs.py 
```

It will print results like
```
snap/vlm/test_finetune/glueepoch_checkpoint-epoch0019
     RTE    MRPC   STS-B    CoLA   SST-2    QNLI     QQP    MNLI MNLI-MM    GLUE
   54.51   84.72   87.18   52.32   90.02   88.36   87.16   81.92   82.57   78.75
snap/vlm/bert_6L_512H_wiki103_sharedheadctr_noshuffle/glueepoch_checkpoint-epoch0029
     RTE    MRPC   STS-B    CoLA   SST-2    QNLI     QQP    MNLI MNLI-MM    GLUE
   58.12   82.76   84.45   26.74   89.56   84.40   86.52   77.56   77.99   74.23
```

### BERT (As baselines)
We also provide pure language-model pre-training as baselines.

#### Wiki103
```shell script
# bash scripts/small_wiki103.bash $GPUs $SNAP_NAME
bash scripts/small_wiki103.bash 0,1,2,3 bert_small
```
It will run a BERT-6Layers-512Hiddens model on [wiki103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)
dataset with the maked language model only.
The snapshot will be saved to `snap/bert/wiki103_bert_small`.

Or you could direclty using the script `small_wiki103_glue.bash` to 
enable GLUE evaluation after finishing pre-training.
```shell script
bash scripts/small_wiki103_glue.bash 0,1,2,3 bert_small
```


