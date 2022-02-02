import os, sys
import argparse
import multiprocessing
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
try:
    from pytorch_lightning.callbacks.lr_logger import LearningRateLogger
except:
    from pytorch_lightning.callbacks import LearningRateMonitor as LearningRateLogger
import torch
from torch.optim.lr_scheduler import LambdaLR
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import PretrainedConfig, AutoTokenizer, AutoModelForMaskedLM
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
from cmed.ed_datasets import WikiDataset
from cmed.pretrain_config import FLAGS, flags
from cmed.utils import (
    CheckpointEveryNSteps, print_parameters, load_kg_embeddings
)
from cmed.kgs.dataset import infiniteloop, Dbpedia
import numpy as np
from cmed.config import FastKGBertConfig
from cmed.models import EntityDisambiguation
from cmed.optimizer import Lamb
from cmed.batch_fn import KG_DataCollatorForLanguageModeling


flags.DEFINE_string('kg_cache_path', '', help='checkpoint name')
flags.DEFINE_string('kg_pretrained_path', '', help='checkpoint name')
flags.DEFINE_string('kg_filename_path', '', help='checkpoint name')
flags.DEFINE_string('kg_data_path', '', help='checkpoint name')
flags.DEFINE_string('kg_name', '', help='kg name')
flags.DEFINE_string('pretrained_name', 'bert-base-cased', help='kg name')

flags.DEFINE_list('datasets', [''], help='kg name')

flags.DEFINE_enum('kg_inject_mode', 'concat', ['concat', 'prepend'], help='concat or prepend')
flags.DEFINE_integer('kg_hidden_layer', 4, help='kg hidden dimension')
flags.DEFINE_integer('kg_hidden_size', 768, help='kg hidden dimension')
flags.DEFINE_integer('kg_batch_size', 1024, help='training data max sequence length')
flags.DEFINE_integer('data_max_length', 512, help='training data max sequence length')

flags.DEFINE_string('spacy_el_path', './el_1m/nlp', help='path to spacy el')
flags.DEFINE_float('margin_weight', 1.0, help='margin loss')
flags.DEFINE_float('lm_weight', 0.5, help='masked language model loss')
flags.DEFINE_float('diversity_weight', -1e-3, help='diversity distance loss')
flags.DEFINE_float('kg_self_regul_weight', 0.0001, help='KG self regularization loss')
flags.DEFINE_float('L2', 7.469e-12, help='margin loss')
flags.DEFINE_float('kg_weight', 1.0, help='kg align weight')
flags.DEFINE_boolean('consistency_mean_loss', True, 'Add KG consistency type loss')
flags.DEFINE_boolean('with_kg', True, 'Train with knowledge graph loss')
flags.DEFINE_boolean('with_mlm', True, 'Train masked language loss')
flags.DEFINE_boolean('load_pretrain', True, 'Load pretrained kg graph')

flags.DEFINE_float('elm_all_probability', 0.2, help='mask entity word tokens : input_ids -> MASK')
flags.DEFINE_float('elm_probability', 0.4, help='masked entity id tokens : input_kgs -> MASK')

flags.DEFINE_boolean('mine_negative_sampling', False, 'mine negative sample first')

FLAGS(sys.argv)


class EntityLinkingLearner(pl.LightningModule):
    def __init__(self, net, total_iterations, looper=None):
        super().__init__()
        self.net = net
        self.total_iterations = total_iterations
        self.log_grad_norm_step = 100
        self.hidden_states = None
        self.looper = looper

    def forward(self, data):
        return self.net(**data)

    def training_step(self, data, batch_idx):

        kg_batch = next(self.looper)

        inputs = {**data, **kg_batch }

        result = self.forward(inputs)

        if self.global_step % 5 == 0:
            tensorboard = self.logger.experiment
            for key, value in result.items():
                tensorboard.add_scalar(key, value.item(), self.global_step)

        # if self.global_step % self.log_grad_norm_step == 0 and not FLAGS.use_amp:
        #     if len(result) == 8:
        #         for hid in result[7]:
        #             hid.retain_grad()
        #         self.hidden_states = result['']

        return {'loss': result['loss'] }

    # def on_after_backward(self):
    #     # example to inspect gradient information in tensorboard
    #     if not FLAGS.use_amp and self.global_step % self.log_grad_norm_step == 0:
    #         if isinstance(self.hidden_states, tuple):
    #             for idx, hidden_states in enumerate(self.hidden_states):
    #                 avg_grad_norm = torch.norm(
    #                         torch.flatten((hidden_states.grad * hidden_states.shape[0]), start_dim=1), 
    #                         p=2, dim=1).mean()
    #                 self.logger.experiment.add_scalar('grad_norm/'+str(idx+1), avg_grad_norm.item(), self.global_step)

    def configure_optimizers(self):

        def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
            def lr_lambda(current_step):
                learning_rate = max(0.0, 1. - (float(current_step) / float(num_training_steps)))
                learning_rate *= min(1.0, float(current_step) / float(num_warmup_steps))
                return learning_rate

            return LambdaLR(optimizer, lr_lambda, last_epoch)

        def get_params_without_weight_decay_ln(named_params, weight_decay):
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
                    'weight_decay': weight_decay,
                },
                {
                    'params': [p for n, p in named_params if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                },
            ]
            return optimizer_grouped_parameters

        optimizer = Lamb(get_params_without_weight_decay_ln(self.net.named_parameters(), weight_decay=0.1), 
            lr=FLAGS.lr, min_trust=0.25, betas=(0.9, 0.999), eps=1e-08)

        self.lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=FLAGS.warmup_step, 
            num_training_steps=self.total_iterations)


        return [optimizer], [{ 'scheduler': self.lr_scheduler, 'name': 'linear_warmup','interval': 'step', }]



if __name__ == '__main__':
    from transformers import BertForMaskedLM
    from transformers import PretrainedConfig, AutoTokenizer

    os.makedirs(FLAGS.name, exist_ok=True)
    with open(os.path.join('./', FLAGS.name, 'flagfile.txt'), 'w') as f:
        f.write(FLAGS.flags_into_string().replace('\n', '  \n'))


    text_tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer_base)
    print('tokenizer graph loaded!')
    kg_model, entity_vocab_size, relation_vocab_size, type_vocab_size, _  = load_kg_embeddings(FLAGS.kg_pretrained_path)

    print('stats ',entity_vocab_size, relation_vocab_size, type_vocab_size)
    datasets = []
    for dataset_filename in FLAGS.datasets:
        wikidata = WikiDataset(dataset_filename, FLAGS.kg_cache_path, tokenizer=text_tokenizer, max_type_ids=type_vocab_size-1)
        datasets.append(wikidata)
    concat = ConcatDataset( datasets )

    # print(wikidata_blink.h5_filename, wikidata_luke.h5_filename)
    data_collator = KG_DataCollatorForLanguageModeling(tokenizer=text_tokenizer, 
        mlm=True, 
        tokenizer_name=FLAGS.pretrained_name,
        mlm_probability=FLAGS.mlm_prob,
        vocab_size=len(text_tokenizer),
        ent_vocab_size=entity_vocab_size,
    )
    train_loader = DataLoader(concat, batch_size=FLAGS.batch_size, 
        num_workers=FLAGS.num_workers, shuffle=True, collate_fn=data_collator)


    dbpedia_dataset = Dbpedia(
        os.path.join(FLAGS.kg_filename_path,'train.txt'), 'train', datasetname='ntee_2014')
    looper = infiniteloop( DataLoader(dbpedia_dataset, batch_size=FLAGS.kg_batch_size,
        num_workers=FLAGS.num_workers, shuffle=True), to_cuda=True)

    total_epoch = int(FLAGS.total_iterations / len(train_loader)) + 20 # just to be safe

    bert_config = FastKGBertConfig(
        ent_vocab_size=entity_vocab_size+2,
        rel_vocab_size=relation_vocab_size,
        type_ent_vocab_size=type_vocab_size,
        margin_weight=FLAGS.margin_weight,
        kg_weight=FLAGS.kg_weight,
        kg_hidden_size=FLAGS.kg_hidden_size,
        kg_hidden_layers=FLAGS.kg_hidden_layer,
        init_layer_num=FLAGS.init_layers if not FLAGS.baseline else FLAGS.num_layers,
        layer_stack_mult=FLAGS.layer_mult  if not FLAGS.baseline else -1,
        layer_drop_prob=FLAGS.layer_dropout,
        pretrained_name=FLAGS.pretrained_name,
        vocab_size=len(text_tokenizer),
        embed_size=FLAGS.embed_dim,
        lm_weight=FLAGS.lm_weight,
        diversity_weight=FLAGS.diversity_weight,
        kg_self_regul_weight=FLAGS.kg_self_regul_weight,
        hidden_size=FLAGS.hidden_dim,
        num_hidden_layers=FLAGS.num_layers,
        num_attention_heads=FLAGS.heads,
        intermediate_size=FLAGS.intermediate_size,
        hidden_act="gelu",
        total_iterations=FLAGS.total_iterations,
        max_position_embeddings=FLAGS.max_length,
        position_bucket_size=64,
        type_vocab_size= 1 if 'roberta' in FLAGS.pretrained_name else 2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=text_tokenizer.pad_token_id,
    )
    print('Train with KG ?', FLAGS.with_kg)
    model = EntityDisambiguation(bert_config, w_kg=FLAGS.with_kg)
    if 'DKGE' in FLAGS.kg_pretrained_path:
        model.knowledge_model = distmult

    pretrained_model = AutoModelForMaskedLM.from_pretrained(FLAGS.pretrained_name)
    model.bert = pretrained_model.roberta
    model.cls = pretrained_model.lm_head

    for name, params in model.bert.named_parameters():
        if 'embeddings' in name:
            params.requires_grad = False
        elif 'encoder.layer' in name:
            layer_num = int(name.split('encoder.layer.', 1)[1].split('.', 1)[0])
            if layer_num < 5:
                params.requires_grad = False

    if FLAGS.load_pretrain:
        print('load pretrain weights')
        model.knowledge_model.load_state_dict(kg_model.state_dict())
    else:
        print('No load pretrain weights')


    if FLAGS.mine_negative_sampling and FLAGS.ckpt and len(FLAGS.ckpt) > 0:
        from cmed.utils import mine_negative_samples
        negative_mine_file = FLAGS.ckpt+'_negative_matrix'
        if os.path.exists(negative_mine_file):
            negative_matrix = torch.load(negative_mine_file, map_location='cpu')
        else:
            weights = torch.load(FLAGS.ckpt, map_location='cpu')
            state_dict = weights['state_dict']
            new_state_dict = {}
            for key, tensor in state_dict.items():
                new_state_dict[key.replace('net.', '')] = tensor
            model.load_state_dict(new_state_dict)
            negative_matrix = mine_negative_samples(model.knowledge_model.ent_embeddings.weight, neg_size=model.neg_window_size)
            torch.save(negative_matrix, negative_mine_file)
        model.negative_matrix = negative_matrix
        print('finish negative mining')


    checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=os.path.join('./', FLAGS.name))
    bert_config.to_file(os.path.join('./', FLAGS.name, 'config'))
    logger = TensorBoardLogger(
        save_dir=os.getcwd(),
        name=FLAGS.name,
    )
    logger.experiment.add_text('hyperparameter', FLAGS.flags_into_string().replace('\n', '  \n'), 0)
    lr_logger = LearningRateLogger(logging_interval='step')

    total_iterations = FLAGS.total_iterations
    module = EntityLinkingLearner(model, total_iterations=FLAGS.total_iterations, looper=looper)

    distributed_backend = None
    if FLAGS.num_gpus > 1:
        distributed_backend = 'ddp'

    trainer = pl.Trainer(
        logger=logger,
        resume_from_checkpoint=FLAGS.ckpt if FLAGS.ckpt and len(FLAGS.ckpt) > 0 else None,
        gpus=FLAGS.num_gpus, max_epochs=total_epoch, 
        distributed_backend=distributed_backend, 
        #plugins='apex_amp',
        checkpoint_callback=checkpoint_callback,
        accumulate_grad_batches=FLAGS.grad_accumulation,
        precision= 16 if FLAGS.use_amp else 32,
        amp_level= 'O1' if FLAGS.use_amp else None,
        callbacks=[
            CheckpointEveryNSteps(
                save_step_frequency=5000,
                prefix=os.path.join(FLAGS.name, FLAGS.model_name),
                total_checkpoint=5,
            ),
            lr_logger,
        ])
    trainer.fit(module, train_loader)
