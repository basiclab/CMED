import os, glob
from torch.utils.data.dataset import Dataset
import numpy as np
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Tuple, Optional
import torch
from modules.dataset import RDF_SUBJECT_NAME, RDF_TYPE_NAME
from .utils import multidimensional_shifting, negsamp_vectorized_bsearch

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
    from modules.kg.utils import generic_data_collate
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from modules.pipelines import PostProcess, Preprocess, load_index2mapping
    entity2id = torch.load('.cache/dbpediav2-s2/entity2id.pt')
    rel2id = torch.load('.cache/dbpediav2-s2/rel2id.pt')

    text_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    sqlite_path = "../wiki_2020/generated"

    preprocess = Preprocess(sqlite_path, entity2id, text_tokenizer)

    
    data_collate_fn = generic_data_collate(text_tokenizer)
    ed_data_collate = ED_Collate(preprocess, data_collate_fn, output_kg_ids=True)
    # AIDA_Conll('../KGERT-v2/datasets/conll2003-el/dev.txt.tmp', entity2id=entity2id, rel2id=rel2id, tokenizer=text_tokenizer)
    dataset = AIDA_Conll('../KGERT-v2/datasets/conll2003-el/dev.txt', max_token_size=470,
        entity2id=entity2id, tokenizer=text_tokenizer, rel2id=rel2id, is_train=True)
    print(dataset[0])
    # AIDA_Conll('../KGERT-v2/datasets/conll2003-el/test.txt.tmp', entity2id=entity2id, rel2id=rel2id, tokenizer=text_tokenizer)
    # dataloader = DataLoader(dataset, batch_size=24, collate_fn=ed_data_collate, shuffle=False)

    # for (batch, _, _) in dataloader:
    #     print(batch['input_ids'].shape)
    #     # print(batch['neg_types'])
    #     # break