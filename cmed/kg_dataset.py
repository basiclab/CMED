import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import defaultdict
import random
import os
import numpy as np
from .utils import negsamp_vectorized_bsearch

class KGTriplet(Dataset):
    def __init__(self, filename, session, cache='.cache'):
        '''
            filename: original data name
            cache: str main data path
            session: train, test, valid data cache name
        '''
        cache_file  = os.path.join(cache, session+'.pt')
        self.session = session

        if os.path.exists(cache_file):
            self.kg_triplet, self.rel_size, self.ent_size, self.type_size = torch.load(cache_file)
            entity2id = torch.load( os.path.join(self.session_dir, 'entity2id.pt'))
            rel2id = torch.load( os.path.join(self.session_dir, 'rel2id.pt'))
            type2id = torch.load( os.path.join(self.session_dir, 'type2id.pt'))

            self.rel_size = max([ id_ for _, id_ in rel2id.items() ])
            self.ent_size = max([ id_ for _, id_ in entity2id.items()])
            self.type_size = max([ id_ for _, id_ in type2id.items()])
        else:
            self.kg_triplet, self.rel_size, self.ent_size, self.type_size = self.preprocess()

            torch.save((self.kg_triplet, self.rel_size, self.ent_size, self.type_size), cache_file)

        self.entity_size = self.ent_size
        self.relation_size = self.rel_size

    def preprocess(self):
        raise NotImplementedError()

    def sample_negative(self, h, r, t):
        replace_head = random.random() > 0.5
        tmp = random.randint(0, self.entity_size-1)
        while tmp == h or tmp == r:
            tmp = random.randint(0, self.entity_size-1)
        if replace_head:
            return tmp, r, t
        return h, r, tmp

    def sample_negative_types(self, t):
        pos_ids = t

        negative_samples = torch.from_numpy(
            negsamp_vectorized_bsearch([], 
                n_items=self.type_size, n_samp=32).flatten()
            )

        return negative_samples


    def __getitem__(self, idx):
        pos_triplet = self.kg_triplet[idx]
        neg_triplet = self.sample_negative(*pos_triplet)
        return pos_triplet, neg_triplet

    def __len__(self):
        return len(self.kg_triplet)    


class Dbpedia(KGTriplet):
    '''
        negative sample 50% either corrupt tail or head
    '''
    type_names = [
            'http://purl.org/dc/terms/subject',
            'http://www.w3.org/1999/02/22-rdf-syntax-ns#type',
    ]

    def preprocess(self, output_type=False):
        kg_triplet = []
        type_triplet = []

        has_entity2id = os.path.exists(os.path.join(self.session_dir, 'entity2id.pt'))
        entity2id = {}
        type2id = {}
        entity2types = defaultdict(list)
        entity2subject = defaultdict(list)

        rel2id = {}

        if has_entity2id:
            entity2id = torch.load( os.path.join(self.session_dir, 'entity2id.pt'))
        if os.path.exists(os.path.join(self.session_dir, 'rel2id.pt')):
            rel2id = torch.load( os.path.join(self.session_dir, 'rel2id.pt'))
        if os.path.exists(os.path.join(self.session_dir, 'type2id.pt')):
            type2id = torch.load( os.path.join(self.session_dir, 'type2id.pt'))

        if os.path.exists(os.path.join(self.session_dir, 'entity2types.pt')):
            entity2types = torch.load( os.path.join(self.session_dir, 'entity2types.pt'))
            entity2subject = torch.load( os.path.join(self.session_dir, 'entity2subject.pt'))

        id2entity = []
        id2rel = []
        id2type = []
        id2tail = []

        with open(self.filename, 'r') as f:
            for line in f.readlines():
                if len(line.strip().split('\t')) < 2:
                    continue
                head, relations, tail = line.strip().split('\t')

                if relations in self.type_names: # ensure only type name in results
                    type_triplet.append((head, relations, tail))
                    if '/resource/' in head and 'Category:' not in head:
                        if relations == self.type_names[0]: 
                            # subject
                            entity2subject[head].append(tail)
                        elif relations == self.type_names[1]: 
                            # type
                            entity2types[head].append(tail)
                        id2entity.append(head)
                        id2type.append(tail)
                    elif '/resource/' in tail and 'Category:' not in tail:
                        if relations == self.type_names[0]: 
                            # subject
                            entity2subject[tail].append(head)
                        elif relations == self.type_names[1]: 
                            # type
                            entity2types[tail].append(head)
                        id2entity.append(tail)
                        id2type.append(head)
                elif 'Category:' not in tail and 'Category:' not in head:
                    id2entity.append(head)
                    id2entity.append(tail)
                    id2rel.append(relations)
                    kg_triplet.append((head, relations, tail))

        for key in entity2types.keys():
            entity2types[key] = list(set(entity2types[key]))

        if (not has_entity2id) or self.merge_entity_id:

            for type_name in self.type_names:
                if type_name not in rel2id:
                    rel2id[type_name] = len(rel2id)

            id2entity = list(set(id2entity))
            id2rel = list(set(id2rel))
            id2type = list(set(id2type))

            for idx, type_ in enumerate(id2type):
                if type_ not in type2id:
                    type2id[type_] = len(type2id)


            for idx, ent in enumerate(id2entity):
                if ent not in entity2id:
                    entity2id[ent] = len(entity2id)

            for idx, rel in enumerate(id2rel):
                if rel not in rel2id:
                    rel2id[rel] = len(rel2id)

            torch.save(entity2id, os.path.join(self.session_dir, 'entity2id.pt'))
            torch.save(type2id, os.path.join(self.session_dir, 'type2id.pt'))
            torch.save(rel2id, os.path.join(self.session_dir, 'rel2id.pt'))

            torch.save(entity2types, os.path.join(self.session_dir, 'entity2types.pt'))
            torch.save(entity2subject, os.path.join(self.session_dir, 'entity2subject.pt'))


        if output_type:
            type_triplets = []
            for head_id, tail_ids in entity2subject.items():
                if len(tail_ids) > self.max_type_size:
                    type_triplets.append( (entity2id[head_id], rel2id[self.type_names[0] ], torch.from_numpy(np.array([ type2id[t] for t in tail_ids ]))  )  )

            for head_id, tail_ids in entity2types.items():
                if len(tail_ids) > self.max_type_size:
                    type_triplets.append( ( entity2id[head_id], rel2id[self.type_names[1] ], torch.from_numpy(np.array([ type2id[t] for t in tail_ids ]))  )  )

        else:
            for idx, triplet in enumerate(kg_triplet):
                head, rel, tail = triplet
                kg_triplet[idx] = ( entity2id[head], rel2id[rel], entity2id[tail] )
        if output_type:
            return kg_triplet, len(rel2id), len(entity2id), len(type2id), type_triplets
        self.entity2id = entity2id
        return kg_triplet, len(rel2id), len(entity2id), len(type2id)

    def get_stats(self):
        print('total obj-sub triplets  ', len(self.kg_triplet))
        print('total obj-type triplets ', len(self.type_triplets))

    def __init__(self, filename, session, cache='.cache', merge_entity_id=False, datasetname = 'fb15k',*args, **kwargs):
        self.max_type_size = 10
        self.merge_entity_id = merge_entity_id
        self.session_dir = os.path.join(cache, datasetname)
        os.makedirs(self.session_dir, exist_ok=True)
        self.filename = filename
        self.session = session
        super(Dbpedia, self).__init__(
            filename, cache=self.session_dir, session=session, *args, **kwargs)
        type_cache_name =  os.path.join(self.session_dir, session+'_type.pt')
        if os.path.exists(type_cache_name):
            self.type_triplets = torch.load(type_cache_name)
        else:
            type_triplets = self.preprocess(True)[-1]
            torch.save(type_triplets, type_cache_name)
            self.type_triplets = type_triplets
        self.entity2id =  torch.load(os.path.join(self.session_dir, 'entity2id.pt'))

    def __getitem__(self, idx):

        pos_triplet = self.kg_triplet[idx]
        neg_triplet = self.sample_negative(*pos_triplet)

        type_idx = random.randint(0, len(self.type_triplets)-1)
        type_triplets = self.type_triplets[type_idx]

        valid_types = type_triplets[2]
        
        negative_type_triplets = self.sample_negative_types(valid_types)[:self.max_type_size ]

        random_idx = torch.randperm(len(valid_types))[:self.max_type_size].flatten()
        # print(random_idx, valid_types)

        replace_head = random.random() > 0.5
        tmp = random.randint(0, self.entity_size-1)
        while tmp == type_triplets[0]:
            tmp = random.randint(0, self.entity_size-1)

        neg_head = type_triplets[0]
        if replace_head:
            neg_head = tmp

        output_types = ( type_triplets[0], type_triplets[1], 
            valid_types[random_idx].long(), negative_type_triplets.long(), neg_head )
        return pos_triplet, neg_triplet, output_types


def infiniteloop(dataloader, to_cuda=True):
    while True:
        for batch in iter(dataloader):
            pos_triplets, neg_triplets, type_triplets = batch
            if to_cuda:
                pos_triplets = [t.cuda() for t in pos_triplets] 
                neg_triplets = [t.cuda() for t in neg_triplets] 
                type_triplets = [t.cuda() for t in type_triplets] 

            yield {
                'kg_pos_triplets': pos_triplets,
                'kg_neg_triplets': neg_triplets,
                'type_triplets': type_triplets,
            }

if __name__ == "__main__":
    # spo_dataset = SPO('kgs/HowNet.spo', 'hownet')
    # spo_dataset = SPO('kgs/Medical.spo', 'medical')
    # spo_dataset = SPO('kgs/CnDbpedia.spo', 'cndbpedia')

    # train must always first!
    # fb15k_dataset = FB15k('kgs/FB15k-237/train.txt', 'train')
    # print(fb15k_dataset.entity_size, fb15k_dataset.relation_size)
    # fb15k_dataset = FB15k('kgs/FB15k-237/test.txt', 'test', merge_entity_id=True)
    # print(fb15k_dataset.entity_size, fb15k_dataset.relation_size)
    # fb15k_dataset = FB15k('kgs/FB15k-237/valid.txt', 'valid', merge_entity_id=True)
    # print(fb15k_dataset.entity_size, fb15k_dataset.relation_size)

    train_dataset = Dbpedia('kgs/ntee/train.txt', 'train', datasetname='ntee_2014', merge_entity_id=True)
    # test_dataset = Dbpedia('kgs/ntee/test.txt', 'test', datasetname='ntee_2014', merge_entity_id=True)
    # valid_dataset = Dbpedia('kgs/ntee/valid.txt', 'valid', datasetname='ntee_2014', merge_entity_id=True)

    # dbpedia_dataset = Dbpedia('kgs/Dbpedia-snapshot1-v3/train.txt', 'train', datasetname='dbpediav2-s2')
    # print(dbpedia_dataset.entity_size, dbpedia_dataset.relation_size, dbpedia_dataset.type_size)
    # dbpedia_dataset = Dbpedia('kgs/Dbpedia-snapshot1-v3/test.txt', 'test', datasetname='dbpediav2-s2', merge_entity_id=True)
    # print(dbpedia_dataset.entity_size, dbpedia_dataset.relation_size, dbpedia_dataset.type_size)
    # dbpedia_dataset = Dbpedia('kgs/Dbpedia-snapshot1-v3/valid.txt', 'valid', datasetname='dbpediav2-s2', merge_entity_id=True)
    # print(dbpedia_dataset.entity_size, dbpedia_dataset.relation_size, dbpedia_dataset.type_size)

    from torch.utils.data import DataLoader
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for (pos, neg, types) in dataloader:
        h, r, t_types, neg_t_types, neg_h = types
        print('h', t_types.shape, 'neg',neg_t_types.shape)
        break
        # print(types[2].shape, types[3].shape, types[-1].shape)