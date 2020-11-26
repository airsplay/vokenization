# Vokenization

PyTorch code for the EMNLP 2020 paper "[Vokenization: Improving Language Understanding with Contextualized, 
Visual-Grounded Supervision](https://arxiv.org/pdf/2010.06775.pdf)" (Hao Tan and Mohit Bansal).

**Outline**
* [Contextualized Cross-Modal Matching](#contextualized-cross-modal-matching-xmatching)
    * [Downloading Image and Captioning Data](#download-image-and-captioning-data)
    * [Model Training](#training-the-cross-modal-matching-model)
    * [Benchmark (Optional)](#benchmarking-cross-modal-matching-models-optional)
* [Vokenization](#vokenization-vokenization)
    * [Downloading Pure-Language Data](#downloading-and-pre-processing-pure-language-data)
    * [Extracting Visual Feature](#extracting-image-features)
    * [Vokenization Process](#the-vokenization-process)
* [Visually-Supervised Language Model](#visually-supervised-language-model-vlm)
    * [VLM Pre-training](#pre-training-with-vlm)
    * [GLUE Evaluation](#glue-evaluation)
    * [MLM Pre-training (as baselines)](#bert-as-baselines)
    
> Note: I recommend to focus on "Wiki103" first and 
> ingore the code blocks related to "English Wikipedia".
> "Eng Wiki" might take too long to complete.

## Installation
```shell script
pip install -r requirements.txt
```

Require python 3.6 + (to support huggingface [transformers](https://github.com/huggingface/transformers)).

## Contextualized Cross-Modal Matching (xmatching)
In this [module](xmatching) (corresponding to Sec 3.2 of the [paper](https://arxiv.org/pdf/2010.06775.pdf)), 
we want to learn a token-image matching model from sentence-image aligned data (i.e., image captioning data).
The model "contextually" measures the relevance between tokens (i.e., words) and images.
The terminology "contextual" emphasize the nature that 
the sentences (the context) are considered
when measuring the token-image relevance score.


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
The model is trained on MS COCO with pairwise hinge loss (details in Sec. 3.2 of the [paper](https://arxiv.org/pdf/2010.06775.pdf)).

Running Commands:
```bash
# Run the cross-modal matching model with single-machine multi-processing distributed training
# "0,1" indicates using the GPUs 0 and 1.
# "bert_resnext" is the name of this snapshot and would be saved at snap/xmatching/bert_resnext
# "--visn resnext101_32x8d" is the vision backbone
# "--lang bert" is the langaugae backbone
# Speed: 20 min ~ 30 min / 1 Epoch, 20 Epochs by default.
bash scripts/run_xmatching.bash 0,1 bert_resnext --visn resnext101_32x8d --lang bert
```
The options `--visn` and `--lang` specify the architecture of the encoder.
Tested options 
```
--visn $VISN_MODEL
VISN_MODEL={resnet18, resnet34, resnet50, resnet101, resnet152, 
            wide_resnet50_2, wide_resnet101_2, resnext101_32x8d (default), ...} 
--lang $LANG_MODEL
LANG_MODEL={bert, roberta, xlnet, bert-large, ...}
```
For visual backbones, the models in [torchvision](https://pytorch.org/docs/stable/torchvision/models.html) are mostly supported.
You might need to handle the last FC layer, because it is written differently in different backbones.
The language backbones are initialized from huggingface [transformers](https://github.com/huggingface/transformers).

> We found that the results with XLNet is pretty low but have not identified 
> the reason. Results of other backbones are similar.

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
1. **Wiki103**. The [wiki103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) dataset
is a seleted subset of English Wikipedia, containing around 100M tokens.
    ```shell script
    bash data/wiki103/get_data_cased.sh
    ```
2. **English Wikipedia**. 
The script to download and process wiki data are modified from [XLM](https://github.com/facebookresearch/XLM).
It will download a 17G file. 
The speed depends on the networking and it usually takes several hours to filter the data.
The process ends with around 2.8B tokens.
    ```shell script
    bash data/wiki/get_data_cased.bash en
    ```
    Note: For *RoBERTa*, it requires an untokenized version of wiki (o.w. the results would be much lower), 
    so please use the following command:
    ```shell script
    bash data/wiki/get_data_cased_untokenized.bash en
    ```
   
> Note: I recommend to focus on "Wiki103" first and 
> ingore the code blocks related to "English Wikipedia".
> "Eng Wiki" might take too long to complete.
   
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
    bash tokenization/tokenize_wiki103_bert.bash 
    ```
2. English Wikipedia (around 3 hours)
    ```shell script
    bash tokenization/tokenize_wiki_bert.bash 
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
We first build a list of universal image indexes with 
[vokenization/create_image_ids.py](vokenization/create_image_ids.py). 
It is used to unify the image ids in different experiments 
thus the feature array stored in hdf5 could be universally indexed.
The image ids are saved under a shared path `LOCAL_DIR` (default to `data/vokenization`)
 defined in [vokenization/common.py](vokenization/common.py).
The image ids are saved under `data/vokenization/images` with format `{IMAGE_SET}_ids.txt`.
We will make sure that all the experiments agree with this meta info,
so that we would not get different indexing in different retrieval experiments.

> Note: The ids created by [create_image_ids.py](vokenization/create_image_ids.py) are only the order of the images.
> The actual images in the dictionary are provided by `extract_keys.bash`, thus is corresponding to the 
> `_paths.txt`, because the `extract_keys` will filter all broken images and non-existing images.

Commands:
```bash
# Step 1, Build image orders.
python vokenization/create_image_ids.py  
```

#### Extracting Image Features

Extract image features regarding the list built above, using code 
[vokenization/extract_vision_keys.py](vokenization/extract_vision_keys.py). 
The code will first read the image ids saved in `data/vokenization/images/{IMAGE_SET}_ids.txt` and locate the images.
The features will be saved under `snap/xmatching/bert_resnext/keys/{IMAGE_SET}.hdf5`.
It finishes within 1 hour.

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

Diversity 
(in [vokenization/evaluate_diversity.py](vokenization/evaluate_diversity.py))
ensures that the same [token type](https://arxiv.org/pdf/1902.06006.pdf)
is mapped to diverse images regarding its context (i.e., the sentence).
Retrieval 
(in [vokenization/evaluate_retrieval.py](vokenization/evaluate_retrieval.py)) 
measures the correspondence of the token and the retrieved images.

We gather these two utils into one script and the command here:
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


Commands
1. Wiki103 (around 1 hour on 4 Titan V)
    ```shell script
    # Note: mp is the abbreviation for "multi-processing"
    # bash scripts/mpvokenize_wiki103.bash $USE_GPUS $SNAP_NAME
    bash scripts/mpvokenize_wiki103.bash 0,1,2,3 bert_resnext
    ```
2. English Wikipedia (around 1 day on 4 Titan V)
    ```shell script
    # bash scripts/mpvokenize_wiki.bash $USE_GPUS $SNAP_NAME
    bash scripts/mpvokenize_wiki.bash 0,1,2,3 bert_resnext
    ```

> The script will call
> [vokenization/vokenize_corpus_mp.py](vokenization/vokenize_corpus_mp.py)
> to vokenize a corpus. 
> The vokenziation happens in [vokenization/vokenization.py](vokenization/vokenization.py) and
> it use [vokenization/indexing.py](vokenization/indexing.py) to do nearest neighbor search
> (based on [faiss](https://github.com/facebookresearch/faiss)).


## Visually-Supervised Language Model (vlm)

### Pre-Training with VLM
As discussed in Sec. 2 of the [paper](https://arxiv.org/pdf/2010.06775.pdf),
we use previous generated vokens to pre-train the model 
with visual supervision.

#### Wiki103 
After the [vokenization process](#the-vokenization-process) of wiki103,
we could run the model with command:
```shell script
# bash scripts/small_vlm_wiki103_glue.bash $GPUs $SNAP_NAME
bash scripts/small_vlm_wiki103.bash 0,1,2,3 wiki103_bert_small
```
It will call 
[vlm/run_vlm_distributed.py](vlm/run_vlm_distributed.py)
and run a BERT-6Layers-512Hiddens model on [wiki103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)
dataset with the support of voken supervisions.
The snapshot will be saved to `snap/vlm/wiki103_bert_small`.
We recommend to run this Wiki103 experiment first since it will finish 
in a reasonable time (20 hours).
The pure BERT pre-training option is also available [later](#bert-as-baselines)
for comparisons.

Note: defautly, the mixed-precision training is not used.
To support the mixed precision pre-training, 
please install the [nvidia/apex](https://github.com/NVIDIA/apex) library with command:
```shell script
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
After that, you could bring back the option `--fp16` and `--fp16_opt_level O2` in 
the script `scripts/small_vlm_wiki103.bash`.
I recommend to use `--fp16_opt_level O2`.
Although the option O2 might be [unstable](https://github.com/NVIDIA/apex/issues/818#issuecomment-639012282),
it saves a lot memory:
the max per-gpu-batch-size is 32 with O1 but 64 with O2.

#### English Wikipedia
After the [vokenization process](#the-vokenization-process) of wiki103,
we could run the model with command:
```shell script
# bash scripts/base_vlm_wiki.bash $GPUs $SNAP_NAME
bash scripts/base_vlm_wiki.bash 0,1,2,3 wiki_bert_base
```
It will run a BERT-12Layers-768Hiddens (same as BERT_BASE) model on the English Wikipedia
dataset with the support of voken supervisions.
The snapshot will be saved to `snap/vlm/wiki_bert_base`.

It takes around 3-5 days on 4 Titan V / GTX 2080
and around 5-7 days to finish in 4 Titan Pascal/T4 cards.
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
The pre-trained snapshots are evaluated by fine-tuning them on the [GLUE](https://gluebenchmark.com/) 
benchmark.
The code are modified from the huggingface [transformers](https://github.com/huggingface/transformers).

Running GLUE evaluation for snapshots from different epochs:
```bash
# bash scripts/run_glue_epochs.bash $GPUS #SNAP_PATH --snaps $NUM_OF_SNAPS                            
bash scripts/run_glue_epochs.bash 0,1,2,3 snap/vlm/wiki103_bert_small --snaps 7                            
```
It will assess 7 snaps using all 0,1,2,3 GPUs. 
Setting `snaps=-1` will assess all checkpoints.
If you just want to evaluate the last (usually the best) snapshot, please use:
```
bash scripts/run_glue_epochs.bash 0 snap/vlm/wiki103_bert_small --snaps 1
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
It will call 
[vlm/run_lm_distributed.py](vlm/run_lm_distributed.py)
and run a BERT-6Layers-512Hiddens model on [wiki103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)
dataset with the masked language model only.
The snapshot will be saved to `snap/bert/wiki103_bert_small`.

Or you could directly using the script `small_wiki103_glue.bash` to 
enable GLUE evaluation after finishing pre-training.
```shell script
bash scripts/small_wiki103_glue.bash 0,1,2,3 bert_small
```

#### English Wikipedia
Command:
```shell script
# bash scripts/base_wiki.bash $GPUs $SNAP_NAME
bash scripts/base_wiki.bash 0,1,2,3 bert_wiki
```

With GLUE evaluation:
```shell script
bash scripts/base_wiki_glue.bash 0,1,2,3 bert_wiki
```

## Pre-processed Data and Pre-trained Models
### Data

Wiki103 (100M tokens)
```
mkdir -p data/wiki103-cased
wget  https://nlp.cs.unc.edu/data/vokenization/wiki103-cased/wiki.test.raw.bert-base-uncased.hdf5 -P data/wiki103-cased
wget https://nlp.cs.unc.edu/data/vokenization/wiki103-cased/wiki.train.raw.bert-base-uncased.hdf5 -P data/wiki103-cased
wget https://nlp.cs.unc.edu/data/vokenization/wiki103-cased/wiki.valid.raw.bert-base-uncased.hdf5 -P data/wiki103-cased
```

Wiki (2800 M tokens)
```
mkdir -p data/wiki-cased
wget  https://nlp.cs.unc.edu/data/vokenization/wiki-cased/en.test.raw.bert-base-uncased.hdf5 -P data/wiki-cased
wget  https://nlp.cs.unc.edu/data/vokenization/wiki-cased/en.train.raw.bert-base-uncased.hdf5 -P data/wiki-cased
wget  https://nlp.cs.unc.edu/data/vokenization/wiki-cased/en.valid.raw.bert-base-uncased.hdf5 -P data/wiki-cased
```

### Models
- BERT (on Wiki): [https://nlp.cs.unc.edu/data/vokenization/bert_12L_768H_wiki.zip](https://nlp.cs.unc.edu/data/vokenization/bert_12L_768H_wiki.zip)
- BERT + VLM (on Wiki): [https://nlp.cs.unc.edu/data/vokenization/vlm_12L_768H_wiki.zip](https://nlp.cs.unc.edu/data/vokenization/vlm_12L_768H_wiki.zip)
- RoBERTa + VLM (on Wiki): [https://nlp.cs.unc.edu/data/vokenization/vlm_roberta_12L_768H_wiki.zip](https://nlp.cs.unc.edu/data/vokenization/vlm_roberta_12L_768H_wiki.zip)

## Reference
If you find our project useful, please cite this paper:
```
@inproceedings{tan2020vokenization,
  title={Vokenization: Improving Language Understanding with Contextualized, 
Visual-Grounded Supervision},
  author={Tan, Hao and Bansal, Mohit},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing},
  year={2020}
}
```

## Acknowledgement
I thank the support from [Bloomberg Data Science Ph.D. Fellowship](https://www.techatbloomberg.com/bloomberg-data-science-ph-d-fellowship/).
We thank ARO-YIP Award W911NF-18-1-0336, DARPA MCS Grant N66001-19-2-4031, and Google Focused Research Award. 
We thank the reviewers and [Yixin Nie](https://easonnie.github.io/) 
and [Jie Lei](https://www.cs.unc.edu/~jielei/)
for their helpful discussions.
Part of the code are built based on huggingface [transformers](https://github.com/huggingface/transformers) and 
facebook [xlm](https://github.com/facebookresearch/XLM) and [faiss](https://github.com/facebookresearch/faiss).

4K3.
