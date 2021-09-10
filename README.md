# CMEM-KG : Condtional Masked Entity Disambiguation

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
python -m cli.dump --dump_file [Wikipedia Dump] --out_file [Output file].db
```

Convert preprocessed files into our pretraining data format

```
python -m cli.preprocess --dump_db [Wikipedia Dump] \
    --tokenizer roberta-base \
    --entity_vocab [Wikipedia Vocab txt file] \
    --output [Dataset output name] <- place this in a cache file
```

### Start pretraining

For more hyper parameters please refer to train.txt

```
python pretrain.py  \
    --flagfile train.txt \
    --datasets= [Dataset output name] \
    --num_gpus=3 \
    --kg_pretrained_path=ntee_transattn/lightning_logs/version_0/checkpoints/step_checkpoint_44_105000.ckpt
```


### Finetuning from checkpoint


Use eval.sh for easy testing ( run finetuning in seeds 1,2,3,4 )

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



