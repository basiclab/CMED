'''
    Core trainer to pretrain knowledge graph embedding
'''
import pytorch_lightning as pl
import torch
import numpy as np
import sys, os
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from cmed.kgs.dataset import Dbpedia
from cmed.kgs.models import TransE, DistMult, diversity_regularization, TransAttnE
from cmed.kgs.utils import evaluate_, evaluate_types
from cmed.kgs.config import FLAGS

FLAGS(sys.argv)


class KGTrainer(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(*batch)

    def training_step(self, batch, batch_nb):
        pos_triplets, neg_triplets, type_triplets = batch

        loss1 = self.model.calculate_loss(pos_triplets, neg_triplets)

        if FLAGS.mean_loss:
            loss2, type_triplet_loss, l2_regularization = self.model.calculate_loss_avg(type_triplets)
            loss = loss1 + (loss2 + type_triplet_loss) * FLAGS.type_weight

            if FLAGS.l2 > 0:
                loss = loss + l2_regularization * FLAGS.l2
        else:
            loss = loss1


        if FLAGS.l2 > 0:
            loss += self.model.regularization(pos_triplets) * FLAGS.l2
        tensorboard_logs = {}
        if FLAGS.self_regul > 0:
            self_regularization = self.model.self_regularization()
            loss = loss + self_regularization * FLAGS.self_regul
            tensorboard_logs['self_regularization'] = self_regularization

        tensorboard_logs['loss'] = loss

        if FLAGS.mean_loss:
            tensorboard_logs['loss/diversity_penalty'] = type_triplet_loss
            tensorboard_logs['loss/type_avg_loss'] = loss2
        tensorboard_logs['loss/entity_loss'] = loss1

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        pos_triplets, neg_triplets, _ = batch

        loss = self.model.calculate_loss(pos_triplets, neg_triplets)

        l2_regularization =  self.model.self_regularization()

        loss = loss + l2_regularization * FLAGS.l2

        if FLAGS.self_regul != 0:
            loss += self.model.regularization(pos_triplets) * FLAGS.self_regul

        type_hits, type_mrr = evaluate_types(self.model, batch)
        hits, mrr, cnt = evaluate_(self.model, batch)

        return {'val_loss': loss, 'cnt': cnt,
            'hits': hits, 'mrr': mrr,
            'type_mrr': type_mrr, 'type_hits': type_hits
        }

    def validation_epoch_end(self, outputs):

        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        log = {'val/loss': val_loss_mean}
        total_cnt = np.sum([ x['cnt']  for x in outputs ])
        log['val/mrr'] = np.sum([ x['mrr']  for x in outputs ]) / total_cnt
        log['val/hit@3'] = np.sum([ x['hits'][2]  for x in outputs ]) / total_cnt
        log['val/hit@10'] = np.sum([ x['hits'][9]  for x in outputs ]) / total_cnt


        log['val/type_mrr'] = np.sum([ x['type_mrr']  for x in outputs ]) / total_cnt
        log['val/type_hit@3'] = np.sum([ x['type_hits'][2]  for x in outputs ]) / total_cnt
        log['val/type_hit@10'] = np.sum([ x['type_hits'][9]  for x in outputs ]) / total_cnt

        return {'val_loss': val_loss_mean, 'log': log}

    def configure_optimizers(self):
        if FLAGS.optimizer == 'sgd':
            optimizer_cls = torch.optim.SGD
        elif FLAGS.optimizer == 'adam':
            optimizer_cls = torch.optim.Adam
        elif FLAGS.optimizer == 'adagrad':
            optimizer_cls = torch.optim.Adagrad
        optimizer = optimizer_cls(self.model.parameters(), lr=FLAGS.lr)

        def get_linear_warmup(optimizer, num_warmup_steps, last_epoch=-1):
            def lr_lambda(current_step):
                learning_rate = min(1.0, float(current_step) / float(num_warmup_steps))
                return learning_rate

            return LambdaLR(optimizer, lr_lambda, last_epoch)

        self.lr_scheduler = get_linear_warmup(optimizer, num_warmup_steps=1000)

        return [optimizer], [{ 'scheduler': self.lr_scheduler, 'name': 'linear_warmup','interval': 'step', }]

def load_dataset_config(dataset_name):
    if dataset_name == 'hownet':
        train_dataset = SPO('kgs/HowNet.spo', 'hownet')
        valid_dataset = SPO('kgs/HowNet.spo', 'hownet')
        return train_dataset, valid_dataset, None
    elif dataset_name == 'medical':
        train_dataset = SPO('kgs/Medical.spo', 'medical')
        valid_dataset = SPO('kgs/Medical.spo', 'medical')
        return train_dataset, valid_dataset, None
    elif dataset_name == 'cndbpedia':
        train_dataset = SPO('kgs/CnDbpedia.spo', 'cndbpedia', train=True)
        valid_dataset = SPO('kgs/CnDbpedia.spo', 'cndbpedia', train=False)
        return train_dataset, valid_dataset, None
    elif dataset_name == 'cndbpedia-small':
        train_dataset = SPO('kgs/CnDbpedia_small.spo', 'cndbpedia', train=True)
        valid_dataset = SPO('kgs/dev_CnDbpedia_small.spo', 'cndbpedia', train=False)
        return train_dataset, valid_dataset, None
    elif dataset_name == 'cndbpedia-mid':
        train_dataset = SPO('kgs/CnDbpedia_mid.spo', 'cndbpedia', train=True)
        valid_dataset = SPO('kgs/dev_CnDbpedia_mid.spo', 'cndbpedia', train=False)
        return train_dataset, valid_dataset, None
    elif dataset_name == 'cnhownet':
        train_dataset = SPO('kgs/CnHowNet.spo', 'cnhownet', train=True)
        valid_dataset = SPO('kgs/dev_CnDbpedia_mid.spo', 'cnhownet', train=False)
        return train_dataset, valid_dataset, None

    elif dataset_name == 'dbpediav2-snapshot1':
        print('load dbpedia v2')
        train_dataset = Dbpedia('kgs/Dbpedia-snapshot1-v3/train.txt', 'train', datasetname='dbpediav2-s2')
        test_dataset = Dbpedia('kgs/Dbpedia-snapshot1-v3/test.txt', 'test', datasetname='dbpediav2-s2', merge_entity_id=True)
        valid_dataset = Dbpedia('kgs/Dbpedia-snapshot1-v3/valid.txt', 'valid', datasetname='dbpediav2-s2', merge_entity_id=True)
        return train_dataset, valid_dataset, test_dataset
    elif dataset_name == 'ntee':
        train_dataset = Dbpedia('kgs/ntee/train.txt', 'train', datasetname='ntee_2014', merge_entity_id=True)
        test_dataset = Dbpedia('kgs/ntee/test.txt', 'test', datasetname='ntee_2014', merge_entity_id=True)
        valid_dataset = Dbpedia('kgs/ntee/valid.txt', 'valid', datasetname='ntee_2014', merge_entity_id=True)
        return train_dataset, valid_dataset, test_dataset
    else:
        train_dataset = Dbpedia(dataset_name+'train.txt', 'train', datasetname='ntee_2014', merge_entity_id=True)
        test_dataset = Dbpedia(dataset_name+'test.txt', 'test', datasetname='ntee_2014', merge_entity_id=True)
        valid_dataset = Dbpedia(dataset_name+'valid.txt', 'valid', datasetname='ntee_2014', merge_entity_id=True)
        return train_dataset, valid_dataset, test_dataset



def model_choice(model_name):
    if model_name == 'transe':
        print('use transe')
        return TransE
    elif model_name == 'transattne':
        return TransAttnE
    elif model_name == 'distmult':
        return DistMult
    return TransE    


if __name__ == "__main__":
    from cmed.utils import CheckpointEveryNSteps
    # dataset = SPO('kgs/HowNet.spo', 'hownet')
    train_dataset, valid_dataset, test_dataset = load_dataset_config(FLAGS.name)
    train = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.train_batch_size, 
        num_workers=FLAGS.num_workers, shuffle=True)

    valid = None
    if valid_dataset != None:
        valid = torch.utils.data.DataLoader(valid_dataset, batch_size=FLAGS.val_batch_size, 
            num_workers=FLAGS.num_workers)
    entity_size, rel_size, type_size = train_dataset.entity_size, train_dataset.relation_size, train_dataset.type_size
    print(entity_size, rel_size, type_size)
    model_class = model_choice(FLAGS.model_name)

    # pad and mask
    model = model_class( entity_size+2, rel_size+1, type_size+1, FLAGS.dimension, 
        p_norm=FLAGS.norm, margin=FLAGS.margin )

    os.makedirs(FLAGS.name, exist_ok=True)
    with open(os.path.join('./', FLAGS.name, 'flagfile.txt'), 'w') as f:
        f.write(FLAGS.flags_into_string().replace('\n', '  \n'))

    model_wrapper = KGTrainer(model)
    trainer = pl.Trainer(gpus=FLAGS.gpus, 
    max_epochs=FLAGS.epochs,
        distributed_backend=FLAGS.backend,
        # precision=16,
        # amp_level='O3',
        num_sanity_val_steps=100,
        accumulate_grad_batches=FLAGS.accum_batch,
        default_root_dir=FLAGS.output_dir,
        check_val_every_n_epoch=5, 
        limit_val_batches=0.02, # limit_val_batches, val_percent_check
        callbacks=[
            CheckpointEveryNSteps(
                save_step_frequency=15000,
                total_checkpoint=5,
            ),
        ])

    trainer.fit(model_wrapper, train, val_dataloaders=valid)
