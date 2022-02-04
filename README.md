# CMEM-KG : Condtional Masked Entity Disambiguation

## Overview of entire setup process

```
(Prepare a list of knowledge graph ids)-[ extract knowledge graph from dbpedia ]->(  knowledge graph embeddings )
                                                                                                  |
                                                                                  (   Start pretraining process )
```

## Download required data

Download data from here : 

https://drive.google.com/file/d/1z9titDxN3DnqX-o5vrlGyA4rNkF6EcB7/view?usp=sharing


Download pretrain checkpoint from here :

https://drive.google.com/file/d/10Rg3GImKIN_rX22kFJTaRmZFmNjcwU4T/view?usp=sharing

Download cache file from here ( evaluation preprocessed data ):


Unzip all files in this project root


## Setup environment

```
pip install -r requirements.txt
```

## Execute script

### Data setup

Convert Wikipedia dump into preprocessed files

```
python -m cli.dump --dump_file enwiki-20220120-pages-articles-multistream.xml.bz2 --out_file roberta.db
```

Convert preprocessed wikipedia into our pretraining data format

```
python -m cli.preprocess --dump_db [Wikipedia Dump] \
    --tokenizer roberta-base \
    --entity_vocab [Wikipedia Vocab txt file] \
    --output [Dataset output name] <- place this in a cache file
```

Optional : you can also append dataset from Facebook to improve its performance



### Start training

1. Knowledge graph pretraining

```
python -m cli.train_kg \
    --flagfile ./resources/kg_training_params.txt \
    --name <path to knowledge graph triplets text> \
    --model_name transe \
    --output_dir <knowledge graph model output path>
```

If you use the provided data sample (data.zip), you should be able to locate the triplet training files under *data/kgs/ntee/*


After training these embeddings you should be able to locate weights under <knowledge graph model output path>/lightning_logs/version_0/checkpoints/

```
<knowledge graph model output path>
    |- lightning_logs
            |- version_0
                    |- checkpoints
```

2. Entity disambiguation training

For more hyper parameters please refer to train.txt

```
python pretrain.py  \
    --flagfile resources/train_params.txt \
    --datasets= [Dataset output name] \
    --num_gpus=3 \
    --kg_pretrained_path=<pretrain weights from step 1>
```

Example

```
python pretrain.py  \
    --flagfile resources/train_params.txt \
    --datasets=wikipedia.h5 \
    --num_gpus=3 \
    --kg_pretrained_path=outputs/kgs/lightning_logs/version_0/checkpoints/step_checkpoint_150_510000.ckpt \
    --kg_cache_path .cache/ntee_2014

```


### Finetuning from checkpoint

Using eval.sh for easy testing ( run finetuning in seeds 1,2,3,4 )

```
python -m cli.ed_finetune \
    --weight_path <Checkpoint name> \
    --config_name <path to config.json>/config \
    --epochs 10 \
    --train_bs 16 \
    --year 2014  \
    --seed 1 \
    --warmup_step 100 \
    --name eval_name \
    --lr 0.000001 \
    --output_path outputs/eval_output
```



