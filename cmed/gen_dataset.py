import os
import json
import pickle
from collections import OrderedDict
from torch.utils.data.dataset import Dataset
import numpy as np
from typing import Any, Callable, Dict, List, NewType, Tuple, Optional
import torch
from tqdm import tqdm

RDF_SUBJECT_NAME = 'http://purl.org/dc/terms/subject'
RDF_TYPE_NAME = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'

from cmed.utils import multidimensional_shifting, negsamp_vectorized_bsearch
from cmed.ed_datasets import ED_Collate,make_boolean_matrix




def conll_fix_entity2id():
    mapping = {
        'FC_Lada-Togliatti_Togliatti': 'FC_Lada-Tolyatti'
    }

def entity_overlapped(entities_ranges, max_size):
    exists = np.array([0]*max_size)
    for span in entities_ranges:
        if sum(exists[span[0]: span[1]]) == 0:
            exists[span[0]: span[1]] = 1
        else:
            return True
    return False


def convert_w_overlap(text, entities_range, tokenizer):
    words = text.split(' ')

    char2word_idx = []
    char2word = [0]*len(words[0])
    
    for idx, word in enumerate(words[1:]):
        char2word += [idx]
        char2word += [idx+1]*len(word)
        
    entities_boundaries = [ -1 ]*len(words)
    
    for idx, entity in enumerate(entities_range):
        e_s, e_e = entity
        w_s, w_e = char2word[e_s], char2word[e_e]
        if w_s == w_e:
            w_e += 1

        entities_boundaries[w_s:w_e] = [ idx ]*(w_e - w_s)

    sections = []
    entity_value = []
    prev = -100
    assert len(words) == len(entities_boundaries)

    # convert text into sections of text
    for idx, type_ in enumerate(entities_boundaries):
        if prev != type_:
            sections.append([words[idx]])
            entity_value.append(type_)
        else:
            sections[-1].append(words[idx])
        prev = type_

    assert len(entity_value) == len(sections)

    output = None

    for idx, seg in enumerate(sections):
        ent_value = entity_value[idx]
        section_text = ' '.join(seg)
        prefix = '' if idx == 0 else ' '
        output_ = tokenizer(prefix+section_text, add_special_tokens=False, )
        if output is None:
            output = output_
            for key in [ 'has_ent_ids', 'input_kgs']:
                output[key] = []
        else:
            for key in output_.keys():
                output[key] += output_[key]

        if ent_value > -1:
            kg_idx = ent_value
            output['input_kgs'] += [ kg_idx ]*len(output_['input_ids'])
            output['has_ent_ids'] += [ 1 ]*len(output_['input_ids'])
        else:
            output['has_ent_ids'] += [ 0 ]*len(output_['input_ids'])
            output['input_kgs'] += [ -1 ]*len(output_['input_ids'])

    kg_placeholder, kg_boolean_matrix_ = make_boolean_matrix(output['input_kgs'])
    return output, kg_boolean_matrix_

def conversion_pipeline(text, entities_ranges, tokenizer, ent_mask_token=386690):
    '''
    text: string
    entities_range: list of tuple with (start position, length of token)
    '''

    def convert(text, entities_range):
        entities_boundaries = [ -1 ]*len(text)

        for idx, entity in enumerate(entities_range):
            e_s, e_e =  entity
            entities_boundaries[e_s:e_e] = [idx]*(e_e - e_s)

        sections = []
        entity_value = []
        prev = -100
        assert len(text) == len(entities_boundaries)

        # convert text into sections of text
        for idx, type_ in enumerate(entities_boundaries):
            if prev != type_:
                sections.append([text[idx]])
                entity_value.append(type_)
            else:
                sections[-1].append(text[idx])
            prev = type_

        assert len(entity_value) == len(sections)

        output = None

        for idx, seg in enumerate(sections):
            ent_value = entity_value[idx]
            section_text = ''.join(seg)
            output_ = tokenizer(section_text, add_special_tokens=False, )
            if output is None:
                output = output_
                for key in [ 'has_ent_ids', 'input_kgs']:
                    output[key] = []
            else:
                for key in output_.keys():
                    output[key] += output_[key]

            if ent_value > -1:
                kg_idx = ent_value
                output['input_kgs'] += [ kg_idx ]*len(output_['input_ids'])
                output['has_ent_ids'] += [ 1 ]*len(output_['input_ids'])
            else:
                output['has_ent_ids'] += [ 0 ]*len(output_['input_ids'])
                output['input_kgs'] += [ -1 ]*len(output_['input_ids'])

        kg_placeholder, kg_boolean_matrix_ = make_boolean_matrix(output['input_kgs'])
        return output, kg_boolean_matrix_

    if entity_overlapped(entities_ranges, len(text)):
        kg_boolean_matrix = []
        for entity_range in entities_ranges:

            output, kg_boolean_matrix_ = convert_w_overlap(text, [entity_range], tokenizer)
            if len(kg_boolean_matrix_) == 0:
                torch.save((text, entities_ranges), 'overlapped_example.pt')

            kg_boolean_matrix.append(kg_boolean_matrix_[0])

        output['kg_boolean_matrix'] = np.array(kg_boolean_matrix)

        if not np.issubdtype(output['kg_boolean_matrix'].dtype , np.number):
            torch.save((text, entities_ranges), 'overlapped_example.pt')

        assert np.issubdtype(output['kg_boolean_matrix'].dtype , np.number)
    else:
        output, kg_boolean_matrix_ = convert(text, entities_ranges)
        output['kg_boolean_matrix'] = kg_boolean_matrix_


    total_entities = len(entities_ranges)

    if total_entities != output['kg_boolean_matrix'].shape[0]:
        print(total_entities, output['kg_boolean_matrix'].shape[0])
        print(entities_ranges)

    assert total_entities == output['kg_boolean_matrix'].shape[0]

    output['kg_attention_mask'] = [1]*total_entities
    output['kg_inputs'] = [ent_mask_token]*total_entities
    output['seq_len'] = total_entities

    output.pop('has_ent_ids', None)
    output.pop('input_kgs', None)
    return output


class PicklePreprocess():


    def __init__(self, dbpedia_entity2id, tokenizer, ent_mask_token=386690):
        self.tokenizer = tokenizer
        self.ent_mask_token = ent_mask_token
        self.dbpedia_entity2id = dbpedia_entity2id

    def process_one(self, text, entities, K=30):
        '''
            text: raw text
            entities: range of mention text [ (start index 1, token length), (start index 2, token length), ...]
        '''
        inputs = conversion_pipeline(text, entities['pos'], self.tokenizer, ent_mask_token=self.ent_mask_token)

        assert len(entities['candidate_index']) == len(entities['dbpedia_url'])
        assert len(entities['candidates'])== len(entities['dbpedia_url'])

        candidates = list(zip(entities['candidate_index'],
            entities['dbpedia_url'],
            entities['candidates']))

        return inputs, candidates

    def __call__(self, text, entities, K=50):

        if isinstance(text, str):
            return self.process_one(text, entities, K=K)
        else:
            inputs, batch_candidates = [], []
            for one_text, one_entities in zip(text, entities):
                input_, candidates = self.process_one(one_text, one_entities, K=K)
                batch_candidates.append(candidates)
                inputs.append(input_)
            return inputs, batch_candidates

def split_data(sentence_text, entities, text_tokenizer=None):
    if len(entities) == 0:
        return []

    words = sentence_text.split(' ')
    half_word_pos = len(words)//2
    first_para, second_para = ' '.join(words[:half_word_pos]), ' '.join(words[half_word_pos:])
    half_pos = len(first_para)
    # find the middle end of sentence end
    dot_pos = [i for i, ltr in enumerate(sentence_text) if ltr in [')', ',', '(', '!', '/'] ]
    cut_off = 0


    all_pos = [ (ent['pos'], ent['end_pos']) for ent in entities ]

    for idx, pos in enumerate(dot_pos[:-1]):
        if dot_pos[idx] > half_pos:
            cut_off = pos
            for ent_pos in all_pos:
                if ent_pos[0] <= pos  <= ent_pos[1]:
                    cut_off = ent_pos[1]+1
            break
    is_half = False
    if cut_off == 0:
        all_pos_half = len(all_pos)//2
        cut_off = all_pos[all_pos_half][1]
        is_half = True

    first_para, second_para = sentence_text[:cut_off], sentence_text[cut_off:]

    if is_half:
        second_para = second_para.strip()


    first_entities, second_entities = [], []
    for entity in entities:
        if entity['end_pos'] <= cut_off:
            first_entities.append(entity)
            assert entity['end_pos'] <= len(first_para)

        else:
            entity['pos'] -= cut_off
            entity['end_pos'] -= cut_off

            if is_half: # strip the first space
                entity['pos'] -= 1
                entity['end_pos'] -= 1

            assert entity['end_pos'] <= len(second_para)

            second_entities.append(entity)

    return [ (first_para, first_entities), (second_para, second_entities)  ]

def format_as_dbpedia(ent_text):
    return '<http://dbpedia.org/resource/'+''.join(ent_text)+'>'


def recursive_split(main_text, entities, text_tokenizer, max_data_size, entity2id, doc_id, level=0):
    results = split_data(main_text, entities)
    dataset = []

    for (text, meta_data) in results:
        tokenized_len = len(text_tokenizer.tokenize(text))

        if tokenized_len > 0 and tokenized_len < max_data_size and len(meta_data):
            labels = []
            kg_ids, pos, candidates, candidate_index, dbpedia_url = [], [], [], [], []
            for meta in meta_data:
                entity_range = (meta['pos'], meta['end_pos'])
                if entity_range not in pos:
                    pos.append(entity_range)
                    candidates.append(meta['candidates'])
                    candidate_index.append(meta['candidate_index'])
                    dbpedia_url.append(meta['candidate_dbpedia'])
                    kg_ids.append(entity2id[format_as_dbpedia(meta['gold'])] )
                    labels.append(meta['gold'][0])

            dataset.append({
                'text': text,
                'kg_ids': kg_ids,
                'entities': {
                    'doc': doc_id,
                    'pos': pos,
                    'candidates': candidates,
                    'candidate_index': candidate_index,
                    'dbpedia_url': dbpedia_url,
                },
                'length': tokenized_len,
                'labels': labels
            })
        elif level == 5 and len(meta_data):
            print('possible overflow, check your device')
            print(tokenized_len, meta_data)
        elif tokenized_len >= max_data_size and len(meta_data):
            dataset += recursive_split(text, meta_data, text_tokenizer, max_data_size, entity2id, doc_id, level+1)

    return dataset

class Generated_REL(Dataset):

    def add_types_relations(self, kg_id):
        subject_types = self.entity2subject[kg_id]
        subject_size = len(subject_types)
        if subject_size > self.type_sample_size:
            subject_types = self.entity2subject[kg_id][multidimensional_shifting(subject_size, 1, np.ones(subject_size)/subject_size).flatten()[:self.type_sample_size]]
        else:
            if len(subject_types) < self.type_sample_size:
                subject_types = np.pad(subject_types, (0, self.type_sample_size-len(subject_types)), 'constant',constant_values=(self.ent_mask_token))

        rdf_types = self.entity2type[kg_id].copy()
        rdf_size = len(rdf_types)

        if rdf_size  > self.type_sample_size:
            rdf_types = self.entity2type[kg_id][multidimensional_shifting(rdf_size, 1, np.ones(rdf_size)/rdf_size).flatten()[:self.type_sample_size]]

        else:
            if len(rdf_types) < self.type_sample_size:
                rdf_types = np.pad(rdf_types, (0, self.type_sample_size-len(rdf_types)), 'constant',constant_values=(self.ent_mask_token))

        neg_rdf_types = negsamp_vectorized_bsearch(self.entity2type[kg_id], self.ent_mask_token,
            items=self.valid_type)[:self.type_sample_size]

        neg_subject_types = negsamp_vectorized_bsearch(self.entity2subject[kg_id], self.ent_mask_token,
            items=self.valid_subject)[:self.type_sample_size]

        return rdf_types, subject_types, neg_rdf_types, neg_subject_types


    def __init__(self, dataset, text_tokenizer,
        entity2id, rel2id, is_train=False,
        ent_mask_token=386690,
        ent_pad_token=386691,
        cache_path='.cache/dbpediav2-s2/',
        max_data_size=500, pickle_path='../wiki_2014/generated/test_train_data/'):
        self.ent_mask_token =ent_mask_token
        self.ent_pad_token = ent_pad_token
        self.entity2subject = torch.load(os.path.join(cache_path, 'entity2subject.pt'))
        self.entity2type = torch.load(os.path.join(cache_path, 'entity2types.pt'))
        self.valid_subject = torch.load(os.path.join(cache_path, 'subject_index.pt'))
        self.valid_type = torch.load(os.path.join(cache_path, 'type_index.pt'))

        self.entity2id = entity2id
        self.rel2id = rel2id
        self.subject_rel_id = rel2id[RDF_SUBJECT_NAME]
        self.type_sample_size = 10
        self.is_train = is_train
        self.type_rel_id = rel2id[RDF_TYPE_NAME]
        self.total_gold = 0
        pickle_file = os.path.join(pickle_path, dataset)
        self.dataset = []

        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)

            fp = open(dataset+'.txt', 'w')

            for doc_id, sentences in data.items():
                # This document is excluded in Le and Titov 2018:
                # https://github.com/lephong/mulrel-nel/blob/db14942450f72c87a4d46349860e96ef2edf353d/nel/dataset.py#L221
                if doc_id == 'Jiří_Třanovský Jiří_Třanovský':
                    continue

                doc = OrderedDict()
                doc_dataset = []

                for sent in sentences:
                    text = sent.pop('sentence')
                    gold_text = sent['gold']
                    self.total_gold += 1

                    if format_as_dbpedia(gold_text) not in entity2id:
                        fp.write(doc_id+','+gold_text[0])
                        continue

                    candidates = sent['candidates']
                    new_candidates = []
                    candidate_index =  []
                    dbpedia_key = []
                    for can in candidates:
                        key = format_as_dbpedia(can[0])
                        if key in entity2id:
                            new_candidates.append(can)
                            candidate_index.append( entity2id[key] )
                            dbpedia_key.append(key)
                        else:
                            fp.write(doc_id+','+gold_text[0])


                    if len(new_candidates) == 0:
                        fp.write(doc_id+','+gold_text[0])
                        continue

                    assert len(new_candidates) > 0

                    sent['candidates'] = new_candidates # p_e_m
                    sent['candidate_index'] = np.array(candidate_index)
                    sent['candidate_dbpedia'] = dbpedia_key # dbpedia_url
                    if text not in doc:
                        doc[text] = [ sent ]
                    else:
                        doc[text].append(sent)

                for text, meta_data in doc.items():
                    tokenized_len = len(text_tokenizer.tokenize(text))
                    if tokenized_len < max_data_size and len(meta_data):
                        for meta in meta_data:
                            text[meta['pos']: meta['end_pos']]
                        labels = []
                        kg_ids, pos, candidates, candidate_index, dbpedia_url = [], [], [], [], []
                        for meta in meta_data:
                            entity_range = (meta['pos'], meta['end_pos'])
                            if entity_range not in pos:
                                pos.append(entity_range)
                                candidates.append(meta['candidates'])
                                candidate_index.append(meta['candidate_index'])
                                dbpedia_url.append(meta['candidate_dbpedia'])
                                labels.append( meta['gold'][0] )
                                kg_ids.append(entity2id[format_as_dbpedia(meta['gold'])] )

                        doc_dataset.append({
                            'text': text,
                            'kg_ids': kg_ids,
                            'entities': {
                                'doc': doc_id,
                                'pos': pos,
                                'candidates': candidates,
                                'candidate_index': candidate_index,
                                'dbpedia_url': dbpedia_url,
                            },
                            'length': tokenized_len,
                            'labels': labels
                        })
                    elif tokenized_len >= max_data_size:
                        doc_dataset += recursive_split(text, meta_data, text_tokenizer, max_data_size, entity2id, doc_id)
                if len(doc_dataset) > 0:
                    self.dataset += [doc_dataset[0]]

                    for doc_idx in range(2, len(doc_dataset)):
                        if (doc_dataset[doc_idx]['length']+self.dataset[-1]['length']+1) < max_data_size:

                            last_data = self.dataset[-1]
                            prev_text_len = len(last_data['text'])

                            last_data['length'] += doc_dataset[doc_idx]['length']
                            last_data['labels'] += doc_dataset[doc_idx]['labels']
                            last_data['text'] += ' '+doc_dataset[doc_idx]['text']
                            last_data['kg_ids'] += doc_dataset[doc_idx]['kg_ids']
                            last_data['entities']['candidates'] += doc_dataset[doc_idx]['entities']['candidates']
                            last_data['entities']['candidate_index'] += doc_dataset[doc_idx]['entities']['candidate_index']
                            last_data['entities']['dbpedia_url'] += doc_dataset[doc_idx]['entities']['dbpedia_url']
                            last_data['entities']['pos'] += [ (  pos[0]+prev_text_len+1, pos[1]+prev_text_len+1 ) for pos in  doc_dataset[doc_idx]['entities']['pos'] ]
                            self.dataset[-1] = last_data
                        else:
                            self.dataset.append(doc_dataset[doc_idx])

            fp.close()

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        output = self.dataset[idx]
        if self.is_train:
            subjects = []
            neg_subjects, neg_types = [], []
            types = []
            for kg_idx in output['kg_ids']:

                type_type, subject, neg_rdf_types, neg_subject_types = self.add_types_relations(kg_idx)
                subjects.append(subject)
                types.append(type_type)
                neg_subjects.append(neg_subject_types)
                neg_types.append(neg_rdf_types)

            output['subjects'] = np.array(subjects)
            output['types'] = np.array(types)
            output['neg_subjects'] = np.array(neg_subjects)
            output['neg_types'] = np.array(neg_types)

            output['subject_rel'] = [self.subject_rel_id] *len(output['kg_ids'])
            output['type_rel'] = [self.type_rel_id] *len(output['kg_ids'])

        return output



def split_docs(data_name, document, entity2id, tokenizer, max_seq_length=412, mode='train', max_candidate_len=30):

    def generate_feature_dict(tokens, mentions, ctx_start, ctx_end, snt_start, snt_end, entity_idx, mode='train'):
        '''
        {
            'text': TEXT,
            'kg_ids': ,
            'entities': {
            'doc': doc id,
            'pos': position,
            'candidates': candidate postfix,
            'candidate_index': candidate index,
            'dbpedia_url' : candidate dbpedia url
            },
            'length': tokenized length,
            'labels':
        }
        '''
        all_tokens = [tokenizer.cls_token] + tokens[ctx_start:ctx_end] + [tokenizer.sep_token]
        texts = tokenizer.convert_tokens_to_string(all_tokens)

        pos = []
        labels = []
        kg_ids = []
        entity_candidates = []
        entity_urls = []
        entity_candidate_idx = []

        for start, end, mention in mentions:
            if start >= snt_start and end <= snt_end:

                candidates = [( '<http://dbpedia.org/resource/'+c['title'].replace(' ','_')+'>',
                                c['title'].replace(' ','_'),
                                c['prior_prob']) for c in mention['candidates'][:max_candidate_len]]

                raw_candidates = [c['title'] for c in mention['candidates'][:max_candidate_len]]+['[NO_E]']
                candidates.append(('[NO_E]', '[NO_E]' , 0.0)) #

                if mode == 'train' and mention['title'] not in raw_candidates:
                    continue

                
                str_start_pos = len(tokenizer.convert_tokens_to_string(all_tokens[ :start - ctx_start+1 ]))
                str_end_pos = len(tokenizer.convert_tokens_to_string(all_tokens[:end- ctx_start+1]))

                if str_end_pos >= len(texts) or str_start_pos >= len(texts):
                    continue

                if str_start_pos >= str_end_pos:
                    continue

                if texts[str_start_pos] == ' ':
                    str_start_pos += 1

                pos.append((str_start_pos, str_end_pos))
                labels.append(mention['title'].replace(' ', '_'))

                entity_key = '<http://dbpedia.org/resource/'+labels[-1]+'>'
                if labels[-1] == '[NO_E]':
                    entity_key = '[NO_E]'
                kg_ids.append( entity2id[entity_key] )

                entity_candidate_idx.append([ entity2id[can[0]] for can in candidates ]  )
                entity_urls.append([ can[0] for can in candidates ]  )
                entity_candidates.append([ can[1] for can in candidates ]  )

        return {
            'text': texts,
            'kg_ids': kg_ids,
            'labels': labels,
            'length': len(tokens[ctx_start:ctx_end])+2,
            'entities': {
                'doc': '-1',
                'pos': pos,
                'candidates': entity_candidates,
                'candidate_index': entity_candidate_idx,
                'dbpedia_url' : entity_urls
            }
        }

    max_num_tokens = max_seq_length - 2
    mention_data = []

    subword_list = [tokenizer.tokenize(' '+w if idx > 0 and w!=str(tokenizer._sep_token)  else w) for idx, w in enumerate(document['words'])] # list of list

    assert len(subword_list) == len(document['words'])
    count = 0
    index_map = {}
    sent_indexes = [] # record end_of_sentence index (in terms of subwords without '[SEP]')

    for i, sub_tokens in enumerate(subword_list):
        index_map[i] = count
        if len(sub_tokens) == 1 and sub_tokens[0] == str(tokenizer._sep_token):
            sent_indexes.append(count)
            continue
        count += len(sub_tokens)

    sub_word_length = count

    sub_word_without_sep = [w for ws in subword_list for w in ws if w != str(tokenizer._sep_token)] # tokenize された '[SEP]' 抜きの subword token 列

    assert sub_word_length == len(sub_word_without_sep)

    if sent_indexes[-1] != sub_word_length: # sub_word_listの最後が[SEP]でなかった場合
        sent_indexes.append(sub_word_length)

    for mention in document['mentions']:
        mention_data.append((index_map[mention['start']], index_map[mention['end']], mention)) # sub_word_withut_sep 上でのindexにmap

    sentence_start = 0
    sentence_record = []
    for sentence_end in sent_indexes:
        sentence_record.append((sentence_start, sentence_end))
        sentence_start = sentence_end

    count = 0
    outputs = []
    for sent_start, sent_end in sentence_record:
        left_token_length = sent_start
        right_token_length = sub_word_length - sent_end
        sentence_length = sent_end - sent_start
        half_context_size = int((max_num_tokens - sentence_length) / 2)

        if left_token_length < right_token_length:
            left_context_length = min(left_token_length, half_context_size)
            right_context_length = min(right_token_length, max_num_tokens - left_context_length - sentence_length)
        else:
            right_context_length = min(right_token_length, half_context_size)
            left_context_length = min(left_token_length, max_num_tokens - right_context_length - sentence_length)
        # sent_start - left_context_length, sent_end + right_context_length, sent_start, sent_end
        entry = generate_feature_dict(sub_word_without_sep, mention_data,
                                  sent_start - left_context_length,
                                  sent_end + right_context_length,
                                  sent_start, sent_end,
                             entity2id, mode=mode)

        if len(entry['labels']) > 0:

            entry['entities']['doc'] = data_name+'_'+str(count)
            outputs.append(entry)
            count += 1
    return outputs

class LUKE_Dataset(Dataset):

    def add_types_relations(self, kg_id):
        subject_types = self.entity2subject[kg_id]
        subject_size = len(subject_types)
        if subject_size > self.type_sample_size:
            subject_types = self.entity2subject[kg_id][multidimensional_shifting(subject_size, 1, np.ones(subject_size)/subject_size).flatten()[:self.type_sample_size]]
        else:
            if len(subject_types) < self.type_sample_size:
                subject_types = np.pad(subject_types, (0, self.type_sample_size-len(subject_types)), 'constant',constant_values=(self.ent_mask_token))

        rdf_types = self.entity2type[kg_id].copy()
        rdf_size = len(rdf_types)

        if rdf_size  > self.type_sample_size:
            rdf_types = self.entity2type[kg_id][multidimensional_shifting(rdf_size, 1, np.ones(rdf_size)/rdf_size).flatten()[:self.type_sample_size]]

        else:
            if len(rdf_types) < self.type_sample_size:
                rdf_types = np.pad(rdf_types, (0, self.type_sample_size-len(rdf_types)), 'constant',constant_values=(self.ent_mask_token))

        neg_rdf_types = negsamp_vectorized_bsearch(self.entity2type[kg_id], self.ent_mask_token,
            items=self.valid_type)[:self.type_sample_size]

        neg_subject_types = negsamp_vectorized_bsearch(self.entity2subject[kg_id], self.ent_mask_token,
            items=self.valid_subject)[:self.type_sample_size]

        return rdf_types, subject_types, neg_rdf_types, neg_subject_types

    def __init__(self, cache_json,
        text_tokenizer,
        entity2id, rel2id, is_train=False,
        ent_mask_token=386690,
        ent_pad_token=386691,
        candidate_size=30,
        max_seq_length=416,
        cache_path='.cache/dbpediav2-s2/',
        data_name='train'):

        self.ent_mask_token = ent_mask_token
        self.ent_pad_token = ent_pad_token
        self.entity2subject = torch.load(os.path.join(cache_path, 'entity2subject.pt'))
        self.entity2type = torch.load(os.path.join(cache_path, 'entity2types.pt'))
        self.valid_subject = torch.load(os.path.join(cache_path, 'subject_index.pt'))
        self.valid_type = torch.load(os.path.join(cache_path, 'type_index.pt'))

        self.entity2id = entity2id
        self.rel2id = rel2id
        self.subject_rel_id = rel2id[RDF_SUBJECT_NAME]
        self.type_sample_size = 10
        self.is_train = is_train
        self.type_rel_id = rel2id[RDF_TYPE_NAME]
        self.total_gold = 0
        entity2id['[NO_E]']

        with open(cache_json, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                docs = data[data_name]
            else:
                docs = data
        self.dataset = []
        for idx, doc in tqdm(enumerate(docs), total=len(docs), dynamic_ncols=True):
            self.dataset += split_docs(data_name+'_'+str(idx), doc, entity2id, text_tokenizer, 
                max_seq_length=max_seq_length, mode='train' if is_train else 'val', 
                max_candidate_len=candidate_size )


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        output = self.dataset[idx]
        if self.is_train:
            subjects = []
            neg_subjects, neg_types = [], []
            types = []
            for kg_idx in output['kg_ids']:

                type_type, subject, neg_rdf_types, neg_subject_types = self.add_types_relations(kg_idx)
                subjects.append(subject)
                types.append(type_type)
                neg_subjects.append(neg_subject_types)
                neg_types.append(neg_rdf_types)

            output['subjects'] = np.array(subjects)
            output['types'] = np.array(types)
            output['neg_subjects'] = np.array(neg_subjects)
            output['neg_types'] = np.array(neg_types)

            output['subject_rel'] = [self.subject_rel_id] *len(output['kg_ids'])
            output['type_rel'] = [self.type_rel_id] *len(output['kg_ids'])

        return output


if __name__ == "__main__":
    from cmed.kgs.utils import generic_data_collate
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    text_tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    entity2id = torch.load('.cache/ntee_2014/entity2id.pt')
    rel2id = torch.load('.cache/ntee_2014/rel2id.pt')
    max_entity_size = 274476
    data_collate_fn = generic_data_collate(text_tokenizer, max_entity_size, max_entity_size+1)

    pickle_path = '../wiki_2014/generated/test_train_data/'
    # pickle_path  ='/mnt/storage/wiki_data/wiki_2019/generated/test_train_data/'

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

    for data_name in [ 'clueweb', 'ace2004', 'aquaint', 'msnbc', 'wikipedia']:
        dataset = LUKE_Dataset('data/cached_2014_v2.json',
            text_tokenizer,
            entity2id, rel2id, is_train=False,
            max_seq_length=400,
            ent_mask_token=max_entity_size,
            ent_pad_token=max_entity_size+1,
            cache_path='.cache/ntee_2014',
            data_name=data_name)
        preprocess = PicklePreprocess(entity2id, text_tokenizer, max_entity_size)
        ed_data_collate = ED_Collate(preprocess, data_collate_fn, output_kg_ids=True, ent_vocab_size=max_entity_size)
        print(data_name)
        print(len(dataset))

        dataloader = DataLoader(dataset, batch_size=24, collate_fn=ed_data_collate, shuffle=False)
        entity_size = -1
        max_seq = -1
        for (batch, candidates, labels, doc_id) in dataloader:
            b, seq_len = batch['input_ids'].shape
            max_seq = max(max_seq, seq_len)
            entity_size = max(entity_size, batch['kg_attention_mask'].shape[0])
        print(entity_size, max_seq)