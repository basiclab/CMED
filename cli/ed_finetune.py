import json
import torch
from torch import nn
import numpy as np
import os, random
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForMaskedLM

from cmed.models import EntityDisambiguation
from cmed.config import FastKGBertConfig
from cmed.ed_datasets import ED_Collate
from cmed.gen_dataset import LUKE_Dataset, PicklePreprocess
from cmed.kgs.utils import generic_data_collate
from cmed.kgs.dataset import infiniteloop, Dbpedia

from .utils import replace_kg
import argparse
import json

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

arg_parser = argparse.ArgumentParser(description='Finetune module')

arg_parser.add_argument('--weight_path', type=str)
arg_parser.add_argument('--config_name', type=str, default='roberta-entity-cmlm-transe/config')
arg_parser.add_argument('--seed', type=int, default=0, help='training seed')
arg_parser.add_argument('--warmup_step', type=int, default=100, help='warmup step')
arg_parser.add_argument('--train_bs', type=int, default=3, help='training batch size')
arg_parser.add_argument('--epochs', type=int, default=30, help='training epochs')
arg_parser.add_argument('--gradient_accumulation', type=int, default=1, help='gradient accumulation')
arg_parser.add_argument('--lr', type=float, default=1e-4, help='training seed')
arg_parser.add_argument('--name', type=str, default='roberta')
arg_parser.add_argument('--year', type=int, default=2019, choices=[2019, 2014])
arg_parser.add_argument("--replace_embed", type=str2bool, nargs='?',
                        default=False,
                        help="Replace embedding with type average")
arg_parser.add_argument("--output_path", type=str, default=None,
                        help="Output results to where")

args = arg_parser.parse_args()

weight_path = args.weight_path
seed = args.seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

config_name = args.config_name
# config_name = 'roberta-entity-linker-byol/config'


warmup_step = args.warmup_step
epochs = args.epochs
gradient_accumulation = args.gradient_accumulation
train_bs = args.train_bs
LR = args.lr

L2_weight = 7e-12
loss_fct = nn.CrossEntropyLoss()

def mse_loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def evaluate(model, dev_dataloader):
    avg_losses = []
    with torch.no_grad():
        for (inputs, candidates, labels, doc_ids) in dev_dataloader:
            kg_ids = inputs.pop('kg_ids')
            kg_ids = kg_ids.cuda()
            for key in inputs.keys():
                inputs[key] = inputs[key].cuda()

            optimizer.zero_grad()
            outputs = model(**inputs)

            masked_position = kg_ids >= 0

            kg_index = kg_ids[masked_position].flatten()

            proj_context_embeddings = outputs['proj_context_embeddings'][masked_position.flatten(), : ]
            proj_head_embeddings = model.knowledge_model.ent_embeddings(kg_index)
            proj_head_embeddings = model.proj(proj_head_embeddings.detach())

            mse_loss = mse_loss_fn( proj_context_embeddings,  proj_head_embeddings).mean()

            avg_losses.append(mse_loss.item())

    return np.mean(avg_losses)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        learning_rate = max(0.0, 1. - (float(current_step) / float(num_training_steps)))
        learning_rate *= min(1.0, float(current_step) / float(num_warmup_steps))
        return learning_rate

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def benchmark_performance(model, dataloader, output_results=False):
    from tqdm import tqdm

    no_candidates = 0
    not_in_kb = 0
    all_entities = 0
    not_in_candidates = 0
    num_gold, num_pred, num_correct, num_mentions = 0, 0, 0, 0

    gold = []
    accuracy = []
    results = []

    try:
        with tqdm(total=len(dataloader), dynamic_ncols=True) as pbar:
            for (inputs, candidates, labels, doc_ids) in dataloader:
                kg_ids = inputs.pop('kg_ids')
                kg_ids = kg_ids.cuda()

                for key in inputs.keys():
                    inputs[key] = inputs[key].cuda()

                with torch.no_grad():
                    outputs = model(**inputs)
                    proj_context_embeddings = outputs['proj_context_embeddings']
                    preds = torch.mm(proj_context_embeddings.view(-1, model.config.kg_hidden_size), model.knowledge_model.ent_embeddings.weight.transpose(0, 1))
                    kg_attention_matrix = inputs['kg_attention_mask']
                    bs, ent_len = kg_attention_matrix.shape
                    preds = preds.view(bs, ent_len, -1)

                for batch_idx in range(len(candidates)):
                    for can_idx in range(len(candidates[batch_idx])):
                        raw_can_idx, dbpedia_urls, p_e_m = candidates[batch_idx][can_idx]
                        wiki_postfix = labels[batch_idx][can_idx]

                        gold.append(wiki_postfix)
                        result = {
                            'id': doc_ids[batch_idx],
                            'candidate': p_e_m,
                            'label': wiki_postfix
                        }

                        if len(dbpedia_urls) == 0:
                            result['pred'] = '[no_candidate]'
                            results.append(result)
                            no_candidates += 1
                            continue


                        dbpedia_urls = np.array(dbpedia_urls)

                        scores = preds[batch_idx][can_idx][ raw_can_idx ]
                        scores = scores.cpu().detach().numpy()
                        pred = dbpedia_urls[np.argsort(-scores)][0]

                        pred_postfix = pred.replace('<http://dbpedia.org/resource/','').replace('>', '')
                        result['pred'] = pred_postfix
                        results.append(result)

                        correct = 0
                        if wiki_postfix != '[NO_E]':
                            num_gold += 1

                        if pred_postfix == wiki_postfix and wiki_postfix != '[NO_E]':
                            correct = 1
                            num_correct += 1

                        if pred_postfix != '[NO_E]':
                            num_pred += 1
                        accuracy.append(correct)

                pbar.update(1)
                pbar.set_postfix({'precision': num_correct/num_pred if num_pred > 0 else 0})

        precision = num_correct/num_pred
        recall = num_correct/num_gold
        f1 = 2*precision * recall /(precision+recall)

        if output_results:
            return f1, precision, recall, results
        return f1, precision, recall
    except AssertionError:
        print('failed')
    return 0, 0


def initialize_parameters(model):

    for name, params in model.entity_decoder.named_parameters():
        if '.LayerNorm' in name:
            params.requires_grad = True
        else:
            params.requires_grad = False

    for name, params in model.knowledge_model.ent_embeddings.named_parameters():
        params.requires_grad = False
    return model

if __name__ == "__main__":
    from REL.db.generic import GenericLookup

    config = FastKGBertConfig.from_file(config_name)
    text_tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_base)



    entity2id = torch.load('.cache/ntee_2014/entity2id.pt')
    id2entity = {}
    not_found = ['[NO_E]']
    max_ent_id = -1
    for ent, ids in entity2id.items():
        max_ent_id = max(ids, max_ent_id)
        if ids not in id2entity:
            id2entity[ids] = ent
        else:
            not_found.append(ent)

    for idx in range(max_ent_id+1):
        if idx not in id2entity:
            key = not_found.pop()
            entity2id[key] = idx
        if len(not_found) == 0:
            break

    rel2id = torch.load('.cache/ntee_2014/rel2id.pt')

    max_entity_size = max([ id_ for _, id_ in entity2id.items() ])+1
    print(max_entity_size)
    data_collate = generic_data_collate(text_tokenizer, max_entity_size, max_entity_size+1)

    preprocess = PicklePreprocess(entity2id, text_tokenizer, max_entity_size)

    ed_data_collate = ED_Collate(preprocess, data_collate, True, ent_vocab_size=max_entity_size)
    ed_train_collate = ED_Collate(preprocess, data_collate, True, mlm_by_seq_len=True, ent_vocab_size=max_entity_size)
    year = args.year
    pickle_path = '../wiki_{}/generated/test_train_data/'.format(year)
    # pickle_path  ='/mnt/storage/wiki_data/wiki_2019/generated/test_train_data/'

    test_datasets = {
            dataset_name: LUKE_Dataset('data/cached_{}_v2.json'.format(year),
                text_tokenizer,
                entity2id, rel2id, is_train=False,
                max_seq_length=450 if dataset_name in ['train', 'test_a', 'test_b'] else 400,
                ent_mask_token=max_entity_size,
                ent_pad_token=max_entity_size+1,
                cache_path='.cache/ntee_2014',
                data_name=dataset_name) \
            for dataset_name in ['train', 'test_a', 'test_b', 'ace2004',
                'aquaint', 'clueweb', 'msnbc', 'wikipedia'
            ]
    }

    print('aida train')
    train_dataset = LUKE_Dataset('data/cached_2014_v2.json',
        text_tokenizer,
        entity2id, rel2id, is_train=True,
        max_seq_length=400,
        ent_mask_token=max_entity_size,
        ent_pad_token=max_entity_size+1,
        cache_path='.cache/ntee_2014',
        data_name='train')

    print('aida A')
    dev_dataset = test_datasets['test_a']
    print('aida testB')
    test_dataset = test_datasets['test_b']

    dbpedia_dataset = Dbpedia('kgs/ntee_transe/train.txt', 'train', datasetname='ntee_2014')
    looper = infiniteloop( DataLoader(dbpedia_dataset, batch_size=512,
        num_workers=10, shuffle=True), to_cuda=True)

    dataloader = DataLoader(train_dataset, batch_size=train_bs, collate_fn=ed_train_collate, shuffle=True, num_workers=4)
    dev_dataloader = DataLoader(dev_dataset, batch_size=8, collate_fn=ed_data_collate, shuffle=False, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=8, collate_fn=ed_data_collate, shuffle=False, num_workers=4)

    model = EntityDisambiguation(config)

    pretrained_model = AutoModelForMaskedLM.from_pretrained(config.tokenizer_base)
    model.bert = pretrained_model.roberta
    model.cls = pretrained_model.lm_head


    model = initialize_parameters(model)


    best_eval_f1 = -1

    weights = torch.load(weight_path)
    state_dict = weights['state_dict']
    new_state_dict = {}
    for key, tensor in state_dict.items():
        new_state_dict[key.replace('net.', '')] = tensor
    model.load_state_dict(new_state_dict)

    if args.replace_embed:
        print('replace knowledge model')
        print('before update', model.knowledge_model.ent_embeddings.weight[0][:10])

        entity2id, relation2id, type2id = torch.load('.cache/ntee_2014/entity2id.pt'),    \
            torch.load('.cache/ntee_2014/rel2id.pt'), \
            torch.load('.cache/ntee_2014/type2id.pt')
        entity2type = torch.load('.cache/ntee_2014/entity2types.pt')
        entity2subject = torch.load('.cache/ntee_2014/entity2subject.pt')
        model.knowledge_model = replace_kg(entity2type, entity2subject, entity2id, relation2id, type2id, model.knowledge_model)
        print('after update', model.knowledge_model.ent_embeddings.weight[0][:10])

    model = model.cuda()
    model.eval()


    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step,
            num_training_steps=int(len(dataloader)*epochs// gradient_accumulation ))

    step = 0
    avg_loss = []

 
    writer = SummaryWriter('logs/'+'{}_{}_{}/{}'.format(year, args.name, weight_path.split('/')[-1], seed) )

    with open('results.jsonl', 'a') as f:
        results_ = {}
        results_['params'] = vars(args)
        results_['scores'] = {}

        for dataset_name in [ 'clueweb', 'msnbc', 'wikipedia',  'train', 'test_a', 
            'test_b', 'ace2004', 'aquaint']:

            if dataset_name not in test_datasets:
                continue

            test_dataset = test_datasets[dataset_name]
            test_dataloader = DataLoader(test_dataset, batch_size=5, 
                collate_fn=ed_data_collate, shuffle=False, num_workers=4)
            outputs = benchmark_performance(model, test_dataloader, 
                output_results= args.output_path != None)
            f1, precision, recall = outputs[0], outputs[1], outputs[2]
            print('{} :\nf1 {:.4f} precision {:.4f}'.format(dataset_name, f1, precision))
            writer.add_scalar('{}/f1'.format(dataset_name.replace('.pkl', '')), f1, step)
            writer.add_scalar('{}/precision'.format(dataset_name.replace('.pkl', '')), precision, step)
            writer.add_scalar('{}/recall'.format(dataset_name.replace('.pkl', '')), recall, step)

            results_['scores']['init_'+dataset_name] = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
            }

            if args.output_path != None and len(outputs) > 2:
                results = outputs[-1]
                with open(os.path.join(args.output_path, 'init_'+dataset_name.replace('.pkl', '')+'.jsonl'), 'w') as g:
                    for row in results:
                        g.write(json.dumps(row)+'\n')

            with open(os.path.join(args.output_path, 'init_result.json'), 'w') as g:
                json.dump(results_, g, indent=4)

        f.write(json.dumps(results_)+'\n')

    f1, precision, recall = benchmark_performance(model, test_dataloader)
    writer.add_scalar('dev_f1', f1, step)
    f1, precision, recall = benchmark_performance(model, dev_dataloader)
    writer.add_scalar('test_f1', f1, step)

    model.train()


    for e in range(epochs):
        with tqdm(total=len(dataloader), dynamic_ncols=True) as pbar:
            pbar.set_description("epoch={}".format(e))
            for (inputs, candidates, labels, doc_ids) in dataloader:

                kg_ids = inputs.pop('kg_ids')

                kg_ids = kg_ids.cuda()
                for key in inputs.keys():
                    inputs[key] = inputs[key].cuda()

                neg_types = inputs.pop('neg_types')
                negative_subjects = inputs.pop('neg_subjects')

                outputs = model(**inputs)
                proj_context_embeddings = outputs['proj_context_embeddings']
                context_embeddings = outputs['context_embeddings']

                subjects =  inputs['subjects']
                subject_rel = inputs['subject_rel']

                non_types_mask = kg_ids != -100

                bs, seq_len, sample_size = subjects.shape

                # softmax
                preds = torch.mm(proj_context_embeddings.view(-1, model.config.kg_hidden_size), 
                    model.knowledge_model.ent_embeddings.weight.transpose(0, 1) )
                mse_loss = loss_fct(preds.view(-1, model.config.ent_vocab_size), kg_ids.view(-1)) / gradient_accumulation

                # ctx margin
                # y = torch.Tensor([-1])
                # if kg_ids.is_cuda:
                #     y = y.cuda()


                # head_embeddings = model.knowledge_model.encode(kg_ids[non_types_mask].flatten()).repeat(1, sample_size, 1)

                # type_context  = context_embeddings[non_types_mask.flatten() ].repeat(1, sample_size, 1)
                # subject_rel_embeddings = model.knowledge_model.extract_rel(
                #     subject_rel[ non_types_mask ].unsqueeze(-1).repeat(1, 1, sample_size)
                # )
                # subject_ids = subjects[non_types_mask, :]
                # negative_subject_ids = negative_subjects[non_types_mask]
                # subject_ent_embeddings = model.knowledge_model.encode(subject_ids)


                # p_sub_score = model.knowledge_model._calc(
                #         type_context.view(-1, model.config.kg_hidden_size) ,
                #         subject_ent_embeddings.view(-1, model.config.kg_hidden_size),
                #         subject_rel_embeddings.view(-1, model.config.kg_hidden_size))


                # corrupt_ent_embeddings = model.knowledge_model.encode(negative_subject_ids)
                # n_sub_score = model.knowledge_model._calc(
                #     type_context.view(-1, model.config.kg_hidden_size),
                #     corrupt_ent_embeddings.view(-1, model.config.kg_hidden_size),
                #     subject_rel_embeddings.view(-1, model.config.kg_hidden_size))


                # types =  inputs['types']
                # type_rel = inputs['type_rel']

                # type_rel_ids = type_rel[ non_types_mask ].unsqueeze(-1).repeat(1, 1, sample_size)
                # type_rel_embeddings = model.knowledge_model.extract_rel(type_rel_ids).view(-1, model.config.kg_hidden_size)
                # type_ent_embeddings = model.knowledge_model.encode(types[non_types_mask, :]).view(-1, model.config.kg_hidden_size)
                # p_rel_score = model.knowledge_model._calc(type_context,
                #     type_ent_embeddings,
                #     type_rel_embeddings).view(-1)

                # negative_type_ids = neg_types[non_types_mask]

                # corrupt_ent_embeddings = model.knowledge_model.encode(negative_type_ids).view(-1, model.config.kg_hidden_size)

                # n_rel_score = model.knowledge_model._calc(type_context,
                #     corrupt_ent_embeddings,
                #     type_rel_embeddings).view(-1)

                # sub_score = model.loss_margin_rank(p_sub_score, n_sub_score, y)
                # rel_score = model.loss_margin_rank(p_rel_score, n_rel_score, y)

                # type_margin_rank_loss = (sub_score.mean() + rel_score.mean())

                # if not torch.isnan(type_margin_rank_loss):
                #     writer.add_scalar('margin_loss', type_margin_rank_loss.item(), step)
                #     mse_loss += type_margin_rank_loss / gradient_accumulation

                mse_loss.backward()
                avg_loss.append(mse_loss.item())

                avg_loss = avg_loss[-10:]
                writer.add_scalar('loss', mse_loss.item()*gradient_accumulation, step)

                if (step+1) % gradient_accumulation == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                step += 1
                pbar.update(1)
                pbar.set_postfix({'loss': '{:.2f}'.format(np.mean(avg_loss)*gradient_accumulation)})

            optimizer.zero_grad()

            model.eval()
            eval_f1, eval_pre, eval_recall = benchmark_performance(model, dev_dataloader)
            if best_eval_f1 < eval_f1:
                print('save checkpoint')
                best_eval_f1 = eval_f1
                state_dict = model.state_dict()
                torch.save(state_dict, 'finetune_aida_e_{}_{}'.format(epochs, weight_path.split('/')[-1]))
            writer.add_scalar('precision', eval_pre, step)
            writer.add_scalar('recall', eval_recall, step)

            writer.add_scalar('f1', eval_f1, step)

            eval_f1, eval_pre, eval_recall = benchmark_performance(model, test_dataloader)
            writer.add_scalar('test_recall', eval_recall, step)
            writer.add_scalar('test_precision', eval_pre, step)
            writer.add_scalar('test_f1', eval_f1, step)

            writer.flush()

            model.train()

    model.eval()

    with open('results.jsonl', 'a') as f:
        results_ = {}
        results_['params'] = vars(args)
        results_['scores'] = {}

        for dataset_name in ['train', 'test_a', 'test_b', 'clueweb', 'test_a', 'test_b', 
            'ace2004', 'aquaint', 'msnbc', 'wikipedia']:

            test_dataset = test_datasets[dataset_name]
            test_dataloader = DataLoader(test_dataset, batch_size=5, 
                collate_fn=ed_data_collate, shuffle=False, num_workers=4)
            outputs = benchmark_performance(model, test_dataloader, 
                output_results= args.output_path != None)

            f1, precision, recall = outputs[0], outputs[1], outputs[2]

            print('{} :\nf1 {:.4f} precision {:.4f}'.format(dataset_name, f1, precision))
            writer.add_scalar('{}/f1'.format(dataset_name.replace('.pkl', '')), f1, step)
            writer.add_scalar('{}/precision'.format(dataset_name.replace('.pkl', '')), precision, step)
            writer.add_scalar('{}/recall'.format(dataset_name.replace('.pkl', '')), recall, step)

            results_['scores'][dataset_name] = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
            }
            if args.output_path != None and len(outputs) > 2:
                results = outputs[-1]
                with open(os.path.join(args.output_path, 'finetune_'+dataset_name.replace('.pkl', '')+'.jsonl'), 'w') as g:
                    for row in results:
                        g.write(json.dumps(row)+'\n')
            with open(os.path.join(args.output_path, 'finetune_result.json'), 'w') as g:
                json.dump(results_, g,  indent=4)
        f.write(json.dumps(results_)+'\n')
    writer.flush()
