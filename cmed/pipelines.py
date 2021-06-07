import torch.nn.functional as F
import numpy as np
import torch
from .ed_dataset import make_boolean_matrix
from .kgs.utils import _calc

def mse_loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)



def load_index2mapping(entity2id, relation2id):
    entity_map = []

    is_category = []
    start_flag = 500000
    end_flag = 0
    entity_map = ['']*len(entity2id)
    relation_map = ['']*len(relation2id)
    is_category = ['']*len(entity2id)
    entity_idxes = []
    type_index = []
    subject_index = []


    for name, idx in entity2id.items():
        entity_map[idx] = name
        if '/Category:' in name:
            subject_index.append(idx)
        elif '/resource/' in name:
            entity_idxes.append(idx)
        else:
            type_index.append(idx)

    for name, idx in relation2id.items():
        relation_map[idx] = name
    return np.array(entity_map), relation_map, np.array(type_index), np.array(entity_idxes), np.array(subject_index)


def conversion_pipeline(text, entities_range, tokenizer, ent_mask_token=386690):
    '''
    text: string
    entities_range: list of tuple with (start position, length of token)
    '''

    entities_boundaries = [-1]*len(text)

    for idx, entity in enumerate(entities_range):
        e_s, e_e =  entity
        entities_boundaries[e_s:e_s+e_e] = [idx]*(e_e)

    output = None
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

    for idx, seg in enumerate(sections):
        ent_value = entity_value[idx]
        section_text = ''.join(seg)
        output_ = tokenizer(section_text, add_special_tokens=False)
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

    kg_placeholder, output['kg_boolean_matrix'] = make_boolean_matrix(output['input_kgs'])
    assert len(kg_placeholder) == len(entities_range)
    output['kg_attention_mask'] = [1]*len(entities_range)
    output['kg_inputs'] = [ent_mask_token]*len(entities_range)
    output['seq_len'] = len(entities_range)

    output.pop('has_ent_ids', None)
    output.pop('input_kgs', None)
    return output



class Preprocess():


    def __init__(self, sqlite_path, dbpedia_entity2id, tokenizer):
        from REL.db.generic import GenericLookup

        self.emb = GenericLookup("entity_word_embedding", save_dir=sqlite_path, table_name="embeddings")
        self.tokenizer = tokenizer

        self.dbpedia_entity2id = dbpedia_entity2id

    def process_one(self, text, entities, K=30):
        '''
            text: raw text
            entities: range of mention text [ (start index 1, token length), (start index 2, token length), ...]
        '''
        inputs = conversion_pipeline(text, entities, self.tokenizer)

        candidates = []
        for (e_s, e_l) in entities:
            mention_text = text[e_s:e_s+e_l]
            candidates.append(self.get_candidates( mention_text, K=K) )
        
        return inputs, candidates

    def __call__(self, text, entities, K=30):

        if isinstance(text, str):
            return self.process_one(text, entities, K=K)
        else:
            inputs, batch_candidates = [], []
            for one_text, one_entities in zip(text, entities):
                input_, candidates = self.process_one(one_text, one_entities, K=K)
                batch_candidates.append(candidates)
                inputs.append(input_)
            return inputs, batch_candidates

    def get_candidates(self, entity_text, K=30):
        p_e_m = self.emb.wiki(entity_text, 'wiki')
        dbpedia_index, dbpedia_key, raw_output = [], [], []
        if p_e_m != None:
            for postfix, conf in p_e_m[:K]:
                dbpedia_url = '<http://dbpedia.org/resource/{}>'.format(postfix)
                if dbpedia_url in self.dbpedia_entity2id:
                    dbpedia_key.append(dbpedia_url)
                    dbpedia_index.append(self.dbpedia_entity2id[dbpedia_url])
                    raw_output.append((postfix, conf))
        return np.array(dbpedia_index), dbpedia_key, raw_output


class PostProcess():

    SUBJECT_NAME = 'http://purl.org/dc/terms/subject'
    RDF_TYPE_NAME = 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type'

    def __init__(self, embeddings, rel_embeddings, 
        ent_index, type_index, subject_index, 
        entity2id, relation2id, entity_map):
        '''

            entity_map : reverse entity2id mapping, map index to original dbpedia link
        '''
        self.subject_rel_idx = relation2id[self.SUBJECT_NAME]
        self.rdf_type_rel_idx = relation2id[self.RDF_TYPE_NAME]

        self.subject_latent = rel_embeddings[self.subject_rel_idx]
        self.subject_len = len(subject_index)
        self.rdf_type_latent = rel_embeddings[self.rdf_type_rel_idx]
        self.rdf_type_len = len(type_index)

        self.entity_embeddings = embeddings[ent_index]
        self.raw_entity_embeddings = embeddings
        self.type_ent_embeddings = embeddings[type_index]
        self.subject_ent_embeddings = embeddings[subject_index]

        self.entity2id = entity2id
        self.relation2id = relation2id

    def __call__(self, context_embeddings, project_embeddings, prior_candidates, 
        type_avg_k=5, subject_avg_k=5):
        '''
            context_embeddings : decoder latent output, one dimension limited only
            project_embeddings : project_embeddings output from decoder, one dimension limited only 
            prior_candidates : numpy list of index mapped to original embedding position
        '''
        candidates_embeddings = self.raw_entity_embeddings[prior_candidates]

        ctx_scores = torch.mm(project_embeddings.view(1, -1), candidates_embeddings.T).flatten()

        if context_embeddings.is_cuda:
            ctx_scores = ctx_scores.cpu()

        ctx_scores = ctx_scores.detach().numpy()
        score_ent_idx = np.argsort(-ctx_scores)

        prior_candidates_mse_rank = prior_candidates[score_ent_idx]

        # reverse query by dct:subject
        subject_context_embeddings = context_embeddings.repeat(self.subject_len, 1)
        subject_rel_embeddings = self.subject_latent.repeat(self.subject_len, 1)
        if subject_context_embeddings.is_cuda:
            subject_context_embeddings = subject_context_embeddings.cpu()
            subject_rel_embeddings = subject_rel_embeddings.cpu()
            self.subject_ent_embeddings = self.subject_ent_embeddings.cpu()
        subject_scores = np.array(_calc(subject_context_embeddings, self.subject_ent_embeddings, subject_rel_embeddings, 1))
        subject_scores_idx = np.argsort(subject_scores)


        avg_subject_ent = self.type_ent_embeddings[subject_scores_idx[:subject_avg_k]].mean(0).repeat(len(prior_candidates), 1)
        type_rel_embeddings = self.subject_latent.repeat(len(prior_candidates), 1)
        if candidates_embeddings.is_cuda:
            candidates_embeddings = candidates_embeddings.cpu()
            avg_subject_ent = avg_subject_ent.cpu()
            type_rel_embeddings = type_rel_embeddings.cpu()
        candidates_subject_scores = np.array(_calc(candidates_embeddings,  avg_subject_ent, type_rel_embeddings, 1))
        candidates_subject_scores_idx = np.argsort(candidates_subject_scores)
        prior_candidates_reverse_subject_rank = prior_candidates[candidates_subject_scores_idx]


        # reverse query by rdf:type
        type_context_embeddings = context_embeddings.repeat(self.rdf_type_len, 1)
        type_rel_embeddings = self.rdf_type_latent.repeat(self.rdf_type_len, 1)
        if type_context_embeddings.is_cuda:
            type_context_embeddings = type_context_embeddings.cpu()
            type_rel_embeddings = type_rel_embeddings.cpu()
            self.type_ent_embeddings = self.type_ent_embeddings.cpu()

        type_scores = np.array(_calc(type_context_embeddings, self.type_ent_embeddings, type_rel_embeddings, 1))
        type_scores_idx = np.argsort(type_scores)
        
        avg_type_ent = self.type_ent_embeddings[type_scores_idx[:type_avg_k]].mean(0).repeat(len(prior_candidates), 1)
        type_rel_embeddings = self.rdf_type_latent.repeat(len(prior_candidates), 1)
        if type_rel_embeddings.is_cuda:
            type_rel_embeddings = type_rel_embeddings.cpu()
            self.type_ent_embeddings = self.type_ent_embeddings.cpu()
        if avg_type_ent.is_cuda:
            avg_type_ent = avg_type_ent.cpu()

        candidates_type_scores = np.array(_calc(candidates_embeddings,  avg_type_ent, type_rel_embeddings, 1))
        candidates_type_scores_idx = np.argsort(candidates_type_scores)
        prior_candidates_reverse_type_rank = prior_candidates[candidates_type_scores_idx]
    

        return {
            'reverse_type_candidate_scores': candidates_type_scores,
            'reverse_subject_candidates_scores': candidates_subject_scores,
            'context_ent_mse_scores': ctx_scores,
            'prior_candidates_mse_rank': prior_candidates_mse_rank,
            'prior_candidates_reverse_type_rank': prior_candidates_reverse_type_rank,
            'prior_candidates_reverse_subject_rank': prior_candidates_reverse_subject_rank,
        }



