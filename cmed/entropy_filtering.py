import math
from collections import defaultdict
from tqdm import tqdm

def entropy(p):
    return p * math.log(p)

def rank_formula1(ent2type, ent2subject):
    adjacency_matrix = {}
    type_freq = defauldict(int)
    for ent, types in ent2type.items():
        if ent not in adjacency_matrix:
            adjacency_matrix[ent] = defaultdict(int)
        for t in types:
            adjacency_matrix[ent][t] += 1
            type_freq[t] += 1
    for ent, subjects in ent2subject.items():
        if ent not in adjacency_matrix:
            adjacency_matrix[ent] = defaultdict(int)
        for s in subjects:
            adjacency_matrix[ent][s] += 1
            type_freq[s] += 1
    

    reverse_matrix = {}
    print('build adjacency matrix')
    for ent, type_ in tqdm(adjacency_matrix.items()):
        denom = len(type_)
        for t in type_:
            if t not in reverse_matrix:
                reverse_matrix[t] = {}
            reverse_matrix[t][ent] = adjacency_matrix[ent][t] / denom

    type_entropies = []
    print('build reverse matrix')
    for type_, entities in reverse_matrix.items():
        type_entropies.append((type_, -sum([ entropy(p) for _, p in entities.items()  ])/len(entities)  ))

    return type_entropies, reverse_matrix, adjacency_matrix, type_freq

def rank_formula2(ent2type, ent2subject):
    '''
        The final formula used in filtering type and subjects
    
    '''
    adjacency_matrix = {}
    type_freq = defaultdict(int)
    
    for ent, types in ent2type.items():
        for t in types:
            if t not in adjacency_matrix:
                adjacency_matrix[t] = defaultdict(int)
            adjacency_matrix[t][ent] += 1
            type_freq[t] += 1
    for ent, subjects in ent2subject.items():
        for s in subjects:
            if s not in adjacency_matrix:
                adjacency_matrix[s] = defaultdict(int)
            adjacency_matrix[s][ent] += 1
            type_freq[s] += 1

    reverse_matrix = {}
    print('build adjacency matrix')
    for type_, ents in tqdm(adjacency_matrix.items()):
        denom = sum([c for _, c in ents.items()])
        if type_ not in reverse_matrix:
            reverse_matrix[type_] = defaultdict(float)
        for e in ents:
            reverse_matrix[type_][e] = adjacency_matrix[type_][e] / denom

    type_entropies = []
    print('build reverse matrix')
    for type_, entities in reverse_matrix.items():
        type_entropies.append((type_, -sum([ entropy(p) for _, p in entities.items()  ])/len(entities)  ))

    return type_entropies, reverse_matrix, adjacency_matrix, type_freq