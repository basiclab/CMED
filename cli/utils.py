import torch
import numpy as np


def replace_kg(entity2type, entity2subject, entity2id, relation2id, type2id, kg_model):
    ent_embedding_weight = kg_model.ent_embeddings.weight.data.clone()

    for ent, idx in entity2id.items():
        types = [ type2id[type_] for type_ in entity2type[ent]] + [ type2id[sub] for sub in entity2subject[ent]]
        rels = [ relation2id['http://www.w3.org/1999/02/22-rdf-syntax-ns#type'] ]*len(entity2type[ent]) + \
            [ relation2id['http://purl.org/dc/terms/subject'] ]*len(entity2subject[ent])
        type_ids = torch.from_numpy(np.array(types)).long()
        rel_ids = torch.from_numpy(np.array(rels)).long()

        output = kg_model.type_embeddings(type_ids)
        rels = kg_model.rel_embeddings(rel_ids)
        new_embed = (rels - output).mean(0).detach()
        ent_embedding_weight[idx] = new_embed

    kg_model.ent_embeddings.weight.data.copy_( ent_embedding_weight )

    return kg_model

if __name__ == "__main__":
    from modules.kge_models import TransAttnE, TransE

    entity2id, relation2id, type2id = torch.load('.cache/ntee_2014/entity2id.pt'),    \
        torch.load('.cache/ntee_2014/rel2id.pt'), \
        torch.load('.cache/ntee_2014/type2id.pt')
    entity2type = torch.load('.cache/ntee_2014/entity2types.pt')
    entity2subject = torch.load('.cache/ntee_2014/entity2subject.pt')

    entity_text = list(entity2type.keys())
    state_dict = torch.load('checkpoints/roberta-entity-cmlm-4/roberta-entity-cmlm-4/roberta-entity-cmlm-4_7_250000.ckpt')['state_dict']
    for key, tensor in state_dict.items():
        if 'knowledge_model' in key:
            new_key = key.replace('model.knowledge_model','')

    model = TransE(entity_vocab_size=len(entity2id), 
        relation_vocab_size=len(relation2id), 
        type_vocab_size=len(type2id), 
        hidden_size=300)
    print(model.ent_embeddings.weight.shape)
    for ent in entity_text:
        types = [ type2id[type_] for type_ in entity2type[ent]] + [ type2id[sub] for sub in entity2subject[ent]]
        type_ids = torch.from_numpy(np.array(types)).long()
        output = model.type_embeddings(type_ids)
        new_embed = output.mean(0)
        print(output.shape)

        break