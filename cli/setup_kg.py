'''

    Setup knowledge graph

    1. Download graph relations from dbpedia : download_graph

    2. Extract required relations and form the final graphs : build_kg_from_output




'''
import glob
import os
from re import sub
import requests
import json
from argparse import ArgumentParser
from tqdm import tqdm
from cmed.constants import (
    DBPEDIA_RSC_PREFIX,
    DBPEDIA_SUBJECT_NAME, DBPEDIA_RDF_TYPE_NAME
)
from cmed.entropy_filtering import rank_formula2

parser = ArgumentParser(description='download knowledge graph from dbpedia')
parser.add_argument('key_list_file', type=str)
parser.add_argument('--json_output_path', type=str, default='dbpedia_jsons')
parser.add_argument('--triplet_output_path', type=str, default='dbpedia_jsons')


def download_graph(entity_key, output_path):
    if '/' in entity_key:
        entity_key = entity_key.replace('/', '')
    output_file = os.path.join(output_path, entity_key+'.json')
    if os.path.exists(output_file):
        return 0

    res = requests.get('http://dbpedia.org/data/{}.json'.format(entity_key), headers={
            'Content-Type': 'application/json',
            'Accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Encoding':'gzip, deflate, br',
            'Accept-Language':'en-GB,en;q=0.5',
            'Cache-Control':'max-age=0',
            'Connection':'keep-alive',
            'Host':'dbpedia.org',
            'Sec-Fetch-Dest':'document',
            'Sec-Fetch-Mode':'navigate',
            'Sec-Fetch-Site':'none',
            'Sec-Fetch-User':'?1',
            'Upgrade-Insecure-Requests':'1',
            'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.14; rv:96.0) Gecko/20100101 Firefox/96.0'
        })
    if res.status_code == 200:
        with open(output_file, 'w') as f:
            json.dump(res.json(), f)
        return 1
    return -1


def build_kg_name(entity_key):
    return DBPEDIA_RSC_PREFIX+entity_key+'>'


def build_kg_from_output(key_list_file, json_output):
    type_relations = set([ DBPEDIA_SUBJECT_NAME, DBPEDIA_RDF_TYPE_NAME ])

    entity2type = {}
    entity2subject = {}
    kg_triplets = []
    valid_entity = []

    with open(key_list_file, 'r') as f:
        for entity_postfix in tqdm(f):
            entity_key = entity_postfix.strip()
            if '/' in entity_key:
                entity_key = entity_key.replace('/', '')
            output_file = os.path.join(json_output, entity_key+'.json')
            if not os.path.exists(output_file):
                continue

            with open(output_file, 'r') as f:
                payload = json.load(f)
            ent_key = build_kg_name(entity_key)
            dbpedia_url = ent_key.replace('<','').replace('>','')
            valid_entity.append(dbpedia_url)

            if dbpedia_url in payload: # expose entity url to the root path
                for key, property in payload[dbpedia_url].items():
                    payload[key] = property

            # build knowledge graph relations from root path
            for key, value in payload.items():
                if isinstance(value, dict) and '/resource/' in key:
                    for p_key, p_value in value.items():
                        if '/property/' in p_key and isinstance(p_value, list):
                            for entity in p_value:
                                if isinstance(entity['value'], str) and '/resource/' in entity['value']:
                                    # triplet property
                                    kg_triplets.append( (key, p_key,  entity['value'] ) )
                # build entity 2 subjects and types
                if key in type_relations:
                    if key == DBPEDIA_RDF_TYPE_NAME:
                        if ent_key not in entity2type:
                            entity2type[ent_key] = []
                        for v in value:
                            if 'value' in v:
                                entity2type[ent_key].append(v['value'])
                        
                    elif key == DBPEDIA_SUBJECT_NAME:
                        if ent_key not in entity2subject:
                            entity2subject[ent_key] = []
                        for v in value:
                            if 'value' in v:
                                entity2subject[ent_key].append(v['value'])
    print('triplets found', len(kg_triplets))
    valid_entity = set(valid_entity)
    # core resource : kg_triplets, entity2type, entity2subject
    final_triplets = []
    for (head, relation, tail ) in kg_triplets:
        if head in valid_entity and tail in valid_entity and head != tail:
            final_triplets.append( ( '<{}>'.format(head), relation, '<{}>'.format(tail)  ) )

    type_entropies, reverse_matrix, adjacency_matrix, type_freq = rank_formula2(entity2type, entity2subject)
    sorted_type_entropies = sorted(type_entropies, key=lambda x:x[1])
    sorted_filter_type_entropies = set([ent  for ent in sorted_type_entropies \
                            if ent[1] >= 0.2 and \
                                    len(reverse_matrix[ent[0]]) > 1 ])

    for entity, subjects in entity2subject.items():
        for subject in subjects:
            if subject in sorted_filter_type_entropies:
                final_triplets.append(  (entity, DBPEDIA_SUBJECT_NAME, '<{}>'.format(subject) ) )
    for entity, types in entity2type.items():
        for type in types:
            if type in sorted_filter_type_entropies:
                final_triplets.append(  (entity, DBPEDIA_RDF_TYPE_NAME, '<{}>'.format(type) ) )

    return final_triplets

def build_triplets_map():

                    if len(line.strip().split('\t')) <= 1:
                        continue
                    head, relations, tail = line.strip().split('\t')
                    if head not in self.entity2id or tail not in self.entity2id:
                        continue

                    if 'DKGE' in self.filename and 'snapshot1' in self.filename or 'dbpedia' in self.filename.lower():
                        rels = [ relations ]
                    else:
                        rels = relations.split('/')

                    for rel in rels:
                        if len(rel) == 0:
                            continue
                        head_id = self.entity2id[head]
                        rel_id = self.rel2id[rel]
                        tail_id = self.entity2id[tail]

                        if head_id not in entity_relations:
                            entity_relations[head_id] = {}
                        if tail_id not in entity_relations:
                            entity_relations[tail_id] = {}
                        
                        if rel_id not in entity_relations[head_id]:
                            entity_relations[head_id][rel_id] = []

                        entity_relations[head_id][rel_id].append(tail_id)
                        
                        if rel_id not in entity_relations[tail_id]:
                            entity_relations[tail_id][rel_id] = []

                        entity_relations[tail_id][rel_id].append(head_id)

        for head_id in entity_relations.keys():
            for rel_id in entity_relations[head_id].keys():
                entity_relations[head_id][rel_id] = list(set(entity_relations[head_id][rel_id]))

        return entity_relations

if __name__ == '__main__':
    args = parser.parse_args()

    if not os.path.exists(args.key_list_file):
        raise ValueError('key_list_file not found')

    # os.makedirs(args.json_output_path, exist_ok=True)

    # with open(args.key_list_file, 'r') as f:
    #     for entity_postfix in tqdm(f):
    #         entity = entity_postfix.strip()
    #         download_graph(entity, args.json_output_path)

    triplets = build_kg_from_output(args.key_list_file, args.json_output_path)
    print(len(triplets), 'found')

