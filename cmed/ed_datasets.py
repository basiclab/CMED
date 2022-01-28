import os
from dataclasses import dataclass
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from typing import Any, Callable, Dict, List, NewType, Tuple
from h5record import H5Dataset, Sequence
from cmed.utils import multidimensional_shifting

RDF_SUBJECT_NAME = 'http://purl.org/dc/terms/subject'
RDF_TYPE_NAME = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'


def make_boolean_matrix(input_kgs):
    input_kgs = np.array(input_kgs)
    seq_len = len(input_kgs)
    position_idx = []
    kg_idx = []
    prev_id = -1
    for idx, kg_id in enumerate(input_kgs):
        if kg_id != -1 and kg_id != prev_id:
            kg_idx.append(kg_id)
            position_idx.append(idx)
        prev_id = kg_id

    boolean_matrix = [ [] ]*len(position_idx)

    for idx, kg_id in enumerate(kg_idx):
        zeros = np.zeros(seq_len)
        mask = (input_kgs == kg_id)
        weight = 1 / mask.sum()
        zeros[mask] = weight
        boolean_matrix[idx] = zeros
    return kg_idx, np.array(boolean_matrix)


class WikiDataset(Dataset):

    def __init__(self, h5_file, cache_path, tokenizer, 
        ent_pad_token_id=0, type_sample_size=10, max_type_ids=97126):
        schema = (
            Sequence('input_ids'),
            Sequence('input_kgs'),
            Sequence('has_ent_ids'),
            Sequence('attention_mask')
        )
        #   tokenizer.knowledgraph.entity_relations
        self.type_sample_size = type_sample_size
        self.max_type_ids = max_type_ids
        self.vocab_size = len(tokenizer)

        self.rel2id = torch.load(os.path.join(cache_path, 'rel2id.pt'))
        ent2id = torch.load(os.path.join(cache_path, 'rel2id.pt'))

        self.subject_rel_id = self.rel2id[RDF_SUBJECT_NAME]
        self.type_rel_id = self.rel2id[RDF_TYPE_NAME]

        self.entity2subject = torch.load(os.path.join(cache_path, 'entity2subject.pt'))
        self.entity2type = torch.load(os.path.join(cache_path, 'entity2types.pt'))
        self.entity_mapping = torch.load(os.path.join(cache_path, 'triplets_map.pt'))

        self.ent_pad_token_id = len(ent2id)-1

        self.data = H5Dataset(schema, h5_file, multiprocess=True)

    def add_types_relations(self, kg_id):
        subject_types = self.entity2subject[kg_id]
        subject_size = len(subject_types)
        if subject_size > self.type_sample_size:
            subject_types = self.entity2subject[kg_id][multidimensional_shifting(subject_size, 1, np.ones(subject_size)/subject_size).flatten()[:self.type_sample_size]]
        else:
            if len(subject_types) < self.type_sample_size:
                subject_types = np.pad(subject_types, (0, self.type_sample_size-len(subject_types)), 'constant',constant_values=(self.max_type_ids))

        rdf_types = self.entity2type[kg_id].copy()
        rdf_size = len(rdf_types)

        if rdf_size  > self.type_sample_size:
            rdf_types = self.entity2type[kg_id][multidimensional_shifting(rdf_size, 1, np.ones(rdf_size)/rdf_size).flatten()[:self.type_sample_size]]
        else:
            if len(rdf_types) < self.type_sample_size:
                rdf_types = np.pad(rdf_types, (0, self.type_sample_size-len(rdf_types)), 'constant',constant_values=(self.max_type_ids))

        return rdf_types, subject_types

    def __getitem__(self, idx):
        row = self.data[idx]

        output = {}
        for key, value in row.items():
            output[key] = value[0]
        output['kg_ids'], output['kg_boolean_matrix'] = make_boolean_matrix(output['input_kgs'])
        tail_ids, rel_ids = [], []
        subjects, types = [], []

        for kg_idx in output['kg_ids']:
            if kg_idx != self.ent_pad_token_id and kg_idx in self.entity_mapping and len(self.entity_mapping[kg_idx]) > 0:
                rel_id =  np.random.choice(list(self.entity_mapping[kg_idx].keys()))
                tail_id =  np.random.choice(self.entity_mapping[kg_idx][rel_id])
                tail_ids.append(tail_id)
                rel_ids.append(rel_id)
                type_type, subject  = self.add_types_relations(kg_idx)
                subjects.append(subject)
                types.append(type_type)

            else:
                tail_ids.append(self.ent_pad_token_id)
                rel_ids.append(self.ent_pad_token_id)
                subjects.append(np.zeros(self.type_sample_size))
                types.append(np.zeros(self.type_sample_size))

        output['subjects'] = np.array(subjects)
        output['types'] = np.array(types)

        output['tail_labels'] = np.array(tail_ids)
        output['rel_labels'] = np.array(rel_ids)
        output['seq_len'] = len(output['kg_ids'])
        output['kg_attention_mask'] = [1]*len(output['kg_ids'])

        output['subject_rel'] = [self.subject_rel_id] *len(output['kg_ids'])
        output['type_rel'] = [self.type_rel_id] *len(output['kg_ids'])

        output.pop('has_ent_ids', None)
        output.pop('input_kgs', None)
        assert max(output['input_ids']) < self.vocab_size
        return output    

    def __len__(self):
        return len(self.data)





class ED_Collate():

    def __init__(self, preprocess, model_collate_fn, output_kg_ids=False, mlm_by_seq_len=False, ent_vocab_size=386689):
        self.collate_fn = model_collate_fn
        self.preprocess = preprocess
        self.ent_vocab_size = ent_vocab_size
        self.output_kg_ids = output_kg_ids
        self.mlm_by_seq_len = mlm_by_seq_len

    def mask_ent(self, kg_ids: torch.Tensor, seq_len: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        inputs = kg_ids.clone()
        labels = kg_ids.clone()
        bs, max_seq_len = labels.shape
        probability_matrix = 1 / seq_len.unsqueeze(1).repeat(1, max_seq_len).float()
        padding_mask = kg_ids.eq(-100)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

        inputs[padding_mask] = self.ent_vocab_size+1 # pad token
        masked_indices = torch.bernoulli(probability_matrix).bool()

        inputs[masked_indices] = self.ent_vocab_size # mask
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        return inputs, labels

    def __call__(self, batch):
        sentence_texts = []
        entities = []
        labels = []
        seq_len = []
        doc_ids = []
        for idx, inputs in enumerate(batch):
            if 'labels' in inputs:
                labels.append(inputs['labels'])
            sentence_texts.append(inputs['text'])
            entities.append(inputs['entities'])
            doc_ids.append(inputs['entities']['doc'])
            seq_len.append(len(inputs['kg_ids']))
        seq_len = torch.from_numpy(np.array(seq_len))

        converted_inputs, candidates = self.preprocess(sentence_texts, entities)
        if self.output_kg_ids:
            for idx in range(len(converted_inputs)):
                converted_inputs[idx]['kg_ids'] = batch[idx]['kg_ids']

                if 'subjects' in batch[idx]:
                    converted_inputs[idx]['subjects'] = batch[idx]['subjects']
                    converted_inputs[idx]['types'] = batch[idx]['types']
                    converted_inputs[idx]['subject_rel'] = batch[idx]['subject_rel']
                    converted_inputs[idx]['type_rel'] = batch[idx]['type_rel']
                    converted_inputs[idx]['neg_subjects'] = batch[idx]['neg_subjects']

                    converted_inputs[idx]['neg_types'] = batch[idx]['neg_types']
        batch = self.collate_fn(converted_inputs)
        if self.output_kg_ids and self.mlm_by_seq_len:
            batch['kg_inputs'], batch['kg_ids'] = self.mask_ent(batch['kg_ids'], seq_len)

        batch.pop('seq_len')
        return batch, candidates, labels, doc_ids

def split_data(sentence_text, entity_range, entity_label, labels):

    words = sentence_text.split(' ')
    half_word_pos = len(words)//2
    first_para, second_para = ' '.join(words[:half_word_pos]), ' '.join(words[half_word_pos:])
    half_pos = len(first_para)
    # find the middle end of sentence end
    dot_pos = [i for i, ltr in enumerate(sentence_text) if ltr in [')', ',', '(', '!', '/'] ]
    cut_off = 0
    for idx, pos in enumerate(dot_pos[:-1]):
        if dot_pos[idx] > half_pos:
            cut_off = pos
            break

    first_para, second_para = sentence_text[:cut_off], sentence_text[cut_off:]
    first_entity_range, second_entity_range = [], []
    first_entity_label, second_entity_label = [], []
    first_labels, second_labels = [], []

    for idx, range_ in enumerate(entity_range):
        start, _ = range_
        if start >= cut_off:
            if (range_[0]-cut_off) < 0:
                print(( range_[0]-cut_off, range_[1] ))
            new_range = ( range_[0]-cut_off, range_[1] )
            second_entity_range.append(new_range)
            second_entity_label.append(entity_label[idx])
            second_labels.append(labels[idx])
        else:
            first_entity_range.append(range_)
            first_entity_label.append(entity_label[idx])
            first_labels.append(labels[idx])
    return ({
            'text': first_para,
            'entities': first_entity_range,
            'entities_text': first_entity_label,
            'labels': first_labels,
        },{
            'text': second_para,
            'entities': second_entity_range,
            'entities_text': second_entity_label,
            'labels': second_labels,
        })



if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer
    from .batch_fn import KG_DataCollatorForLanguageModeling

    # from modules.kgs.utils import generic_data_collate
    from torch.utils.data import DataLoader
    # from modules.pipelines import PostProcess, Preprocess, load_index2mapping
    # entity2id = torch.load('.cache/dbpediav2-s2/entity2id.pt')
    # rel2id = torch.load('.cache/dbpediav2-s2/rel2id.pt')

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    # sqlite_path = "../wiki_2020/generated"

    # preprocess = Preprocess(sqlite_path, entity2id, text_tokenizer)

    
    # data_collate_fn = generic_data_collate(text_tokenizer)
    # ed_data_collate = ED_Collate(preprocess, data_collate_fn, output_kg_ids=True)
    # AIDA_Conll('../KGERT-v2/datasets/conll2003-el/dev.txt.tmp', entity2id=entity2id, rel2id=rel2id, tokenizer=text_tokenizer)
    collate_fn = KG_DataCollatorForLanguageModeling(tokenizer, tokenizer_name='roberta-base')
    dataset = WikiDataset('../fast-transformer/cache/wiki_roberta-base3274405_2603.h5'
        ,'../fast-transformer/.cache/ntee_2014/'
        ,tokenizer)

    print(dataset[0])
    from tqdm import tqdm
    dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn, shuffle=True, num_workers=10)
    for batch in tqdm(dataloader):
        batch

    # AIDA_Conll('../KGERT-v2/datasets/conll2003-el/test.txt.tmp', entity2id=entity2id, rel2id=rel2id, tokenizer=text_tokenizer)
    # dataloader = DataLoader(dataset, batch_size=24, collate_fn=ed_data_collate, shuffle=False)

    # for (batch, _, _) in dataloader:
    #     print(batch['input_ids'].shape)
    #     # print(batch['neg_types'])
    #     # break