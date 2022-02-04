from functools import lru_cache
import math
from torch import nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertOnlyMLMHead, BertEncoder, BertModel
from transformers.models.roberta.modeling_roberta import RobertaLMHead, RobertaModel
from axial_positional_embedding import AxialPositionalEmbedding
import torch
from cmed.kgs.models import TransE, TransAttnE, DistMult
from cmed.kgs.models import diversity_regularization, pairwise_diversity_regularization
from transformers.modeling_outputs import (
    BaseModelOutput
)

def default(val, default_val):
    return default_val if val is None else val


class EntityProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, int(input_dim*3) ),
            nn.GELU(),
            nn.LayerNorm(int(input_dim*3)),
            nn.Linear(int(input_dim*3), output_dim)
        )

    def forward(self, hidden_states):
        output_logits = self.proj(hidden_states)
        return output_logits


class EntityFlow(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        axial_position_shape = default(None, (math.ceil(config.max_position_embeddings / config.entity_bucket_size), config.entity_bucket_size))

        self.position_embeddings = AxialPositionalEmbedding(config.hidden_size, axial_position_shape)


        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.kg_hidden_layers = config.kg_hidden_layers
        config.num_hidden_layers = config.kg_hidden_layers
        config.is_decoder = True
        config.add_cross_attention = True

        self.encoder = BertEncoder(config)
        assert len(self.encoder.layer) == config.kg_hidden_layers

        self.init_weights()

    @staticmethod
    @lru_cache(maxsize=10000)
    def make_entity_embedding(shape):
        return torch.zeros(shape).long()

    def forward(self, 
        embedding,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        encoder_attention_mask=None,
        past_key_values=None,
        output_hidden_states=False,
        encoder_hidden_states=None,
        return_dict=False,
    ):

        pos_embedding = self.position_embeddings(embedding)

        embedding = self.LayerNorm(embedding + pos_embedding)

        input_shape = embedding.size()[:-1]
        batch_size, seq_length = input_shape
        device = embedding.device
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        if encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None


        encoder_outputs = self.encoder(
            embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        sequence_output = encoder_outputs[0]

        return BaseModelOutput(
                last_hidden_state=sequence_output,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
        )


class ZeroShotEntityDisambiguation(BertPreTrainedModel):
    def __init__(self, config, negative_matrix=None, w_kg=True, w_mlm=True):
        super().__init__(config)
        self.entity_decoder = EntityFlow(config)
        self.neg_window_size = 10
        self.w_kg = w_kg
        self.w_mlm = w_mlm
        self.bert = BertModel(config)
        if 'roberta' in config.pretrained_name:
            self.cls = RobertaLMHead(config)
        else:
            self.cls = BertOnlyMLMHead(config)
        self.kg_upsample = nn.Linear(config.kg_hidden_size, config.hidden_size)
        self.kg_downsample = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size*3),
                nn.LayerNorm(config.hidden_size*3),
                nn.GELU(),
                nn.Linear(config.hidden_size*3, config.kg_hidden_size),
        )
        self.ctx_proj = EntityProjection(config.hidden_size, config.kg_hidden_size)
        self.loss_margin_rank = nn.MarginRankingLoss(1, reduction='none')
        self.knowledge_model = TransAttnE(config.ent_vocab_size, 
            config.rel_vocab_size, 
            config.type_ent_vocab_size,
            config.kg_hidden_size)
        self.init_weights()



class EntityDisambiguation(BertPreTrainedModel):
    def __init__(self, config, negative_matrix=None, w_kg=True, w_mlm=True):
        super().__init__(config)
        self.negative_matrix = negative_matrix

        self.entity_decoder = EntityFlow(config)
        self.neg_window_size = 10
        self.w_kg = w_kg
        self.w_mlm = w_mlm
        self.bert = BertModel(config)
        if 'roberta' in config.pretrained_name:
            self.cls = RobertaLMHead(config)
        else:
            self.cls = BertOnlyMLMHead(config)
        self.kg_upsample = nn.Linear(config.kg_hidden_size, config.hidden_size)
        self.kg_downsample = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size*3),
                nn.LayerNorm(config.hidden_size*3),
                nn.GELU(),
                nn.Linear(config.hidden_size*3, config.kg_hidden_size),
        )

        self.ctx_proj = EntityProjection(config.hidden_size, config.kg_hidden_size)


        self.loss_margin_rank = nn.MarginRankingLoss(1, reduction='none')

        self.knowledge_model = TransAttnE(config.ent_vocab_size, 
            config.rel_vocab_size, 
            config.type_ent_vocab_size,
            config.kg_hidden_size)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def tie_weights(self):
        pass

    def forward(
        self,
        input_ids=None,
        kg_inputs=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        kg_boolean_matrix=None,
        kg_attention_mask=None,

        labels=None,
        kg_ids=None,
        kg_labels=None,
        rel_labels=None,
        tail_labels=None,
        kg_pos_triplets=None,
        kg_neg_triplets=None,
        type_triplets=None,

        type_rel=None,        
        types=None,
        subject_rel=None,
        subjects=None,

        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        **kwargs
    ):
        r"""
            labels (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `optional`):
                Labels for computing the masked language modeling loss.
                Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
                in ``[0, ..., config.vocab_size]``
            kg_ids: (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `optional`):
            

            kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
                Used to hide legacy arguments that have been deprecated.

            Returns:

            Example:

                >>> from transformers import BertTokenizer, BertForPreTraining
                >>> import torch

                >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                >>> model = BertForPreTraining.from_pretrained('bert-base-uncased', return_dict=True)

                >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
                >>> outputs = model(**inputs)

                >>> prediction_logits = outputs.prediction_logits
                >>> seq_relationship_logits = outputs.seq_relationship_logits
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )
        sequence_output = outputs[0]
        # print('inputs ', input_ids.max(), input_ids.min(), input_ids.shape)
        encoder_hidden_states = outputs[0]
        first_input_embeds = self.bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds
        )

        embedding = torch.einsum('bks,bsh->bkh', kg_boolean_matrix, first_input_embeds)
        # print('kg_inputs ', kg_inputs.max(), kg_inputs.min())
        ent_embedding = self.kg_upsample(self.knowledge_model.ent_embeddings(kg_inputs))

        entity_outputs = self.entity_decoder(
            embedding=embedding + ent_embedding,
            attention_mask=kg_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        entity_output_sequence = entity_outputs[0]


        prediction_scores = self.cls(sequence_output)

        total_loss, gen_acc, masked_kg_loss = 0, None, None
        losses = []
        results = {}
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_kg_mse = nn.MSELoss()
            loss_bce = nn.BCEWithLogitsLoss()
            loss_cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            if self.w_mlm:
                gen_predictions = torch.argmax(prediction_scores, dim=-1)
                gen_acc = (labels == gen_predictions).float().mean()

                masked_percentage = (labels != -100).sum() / (input_ids != self.config.pad_token_id ).sum()
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
                total_loss += masked_lm_loss * self.config.lm_weight

                results['lm/mlm_loss'] = masked_lm_loss
                results['lm/acc'] = gen_acc
                results['lm/masked_percentage'] = masked_percentage

            non_pad_mask = (kg_ids != -100) & ( rel_labels != -1)

            if non_pad_mask.sum() > 0:
                # attention mask loss
                loss_kg_attn_fct = nn.NLLLoss()
                loss_kg_attn = torch.zeros(1)

            if self.w_kg:
                tails = tail_labels[ non_pad_mask ]
                rels = rel_labels[ non_pad_mask ]
                # print('tails ',tails.max(), tails.max())
                # print('rels ',rels.max(), rels.min())
                tail_embeddings = self.knowledge_model.encode(tails)
                rel_embedding = self.knowledge_model.extract_rel( rels )

                rel_embedding = rel_embedding.view(-1, self.config.kg_hidden_size)
                tail_embeddings = tail_embeddings.view(-1, self.config.kg_hidden_size)


                # Decoder TransE loss
                entity_feature = self.kg_downsample(entity_output_sequence)
                context_embeddings = entity_feature[non_pad_mask].view(-1, self.config.kg_hidden_size)

                p_score = self.knowledge_model._calc( context_embeddings, tail_embeddings, rel_embedding)
                random_entities = torch.randint(high=self.config.ent_vocab_size-1 , size=tails.size())
                if kg_ids.is_cuda:
                    random_entities = random_entities.cuda()

                corrupt_embeddings = self.knowledge_model.encode(random_entities)
                n_score = self.knowledge_model._calc( context_embeddings, corrupt_embeddings, rel_embedding)
                

                regularization_weight = self.knowledge_model.regularization(  (  random_entities ,rels, tails) )
                margin_rank_loss = self.knowledge_model.criterion(p_score, n_score)
                relation_diversity = diversity_regularization(rel_embedding)
                results['kg/context_margin_loss'] = margin_rank_loss
                results['kg/context_regularization'] = regularization_weight
                results['kg/context_rel_regularization'] = relation_diversity

                total_loss += margin_rank_loss * 2
                    # relation_diversity * (self.config.diversity_weight) + \
                    # regularization_weight * self.config.kg_self_regul_weight

            if subject_rel is not None and self.w_kg:

                # print(subject_rel.shape, types.shape)
                # print(subjects.shape, types.shape)
                # print(type_rel.shape, )

                non_types_mask = (subject_rel != -100) & (type_rel != -100)

                non_samples_mask = subjects != -100

                bs, seq_len, sample_size = subjects.shape

                type_context  = entity_feature[non_types_mask].view(-1, self.config.kg_hidden_size).unsqueeze(1).repeat(1, sample_size, 1)
                # print('subject_rel ', subject_rel.max(), subject_rel.min())

                subject_rel_embeddings = self.knowledge_model.extract_rel(
                    subject_rel[ non_types_mask ].unsqueeze(-1).repeat(1, 1, sample_size)
                )
                subject_ids = subjects[non_samples_mask]
                # print('subject_ids ', subject_ids.max(), subject_ids.min())

                subject_ent_embeddings = self.knowledge_model.type_embeddings(subject_ids)

                p_sub_score = self.knowledge_model._calc(type_context.view(-1, self.config.kg_hidden_size) , subject_ent_embeddings, subject_rel_embeddings.view(-1, self.config.kg_hidden_size))

                random_entities = torch.randint(high=self.config.type_ent_vocab_size-1 , size=subject_ids.size())
                if kg_ids.is_cuda:
                    random_entities = random_entities.cuda()
                # print('random_entities 2 ', random_entities.max(), random_entities.min())

                corrupt_ent_embeddings = self.knowledge_model.type_embeddings(random_entities)
                n_sub_score = self.knowledge_model._calc(
                    type_context.view(-1, self.config.kg_hidden_size), corrupt_ent_embeddings.view(-1, self.config.kg_hidden_size), subject_rel_embeddings.view(-1, self.config.kg_hidden_size))

                type_ids = type_rel[ non_types_mask ].unsqueeze(-1).repeat(1, 1, sample_size)
                # print('type_ids ', type_ids.max(), type_ids.min(), type_ids.shape)

                type_rel_embeddings = self.knowledge_model.extract_rel(type_ids)                

                # print('types ', types.max(), types.min(), types.shape)

                type_ent_embeddings = self.knowledge_model.type_embeddings(types[non_samples_mask])

                p_rel_score = self.knowledge_model._calc(
                    type_context.view(-1, self.config.kg_hidden_size), 
                    type_ent_embeddings.view(-1, self.config.kg_hidden_size), 
                    type_rel_embeddings.view(-1, self.config.kg_hidden_size)).view(-1)

                random_entities = torch.randint(high=self.config.type_ent_vocab_size-1 , size=type_ids.size())
                if kg_ids.is_cuda:
                    random_entities = random_entities.cuda()
                # print('random_entities 3 ', random_entities.max(), random_entities.min())

                corrupt_ent_embeddings = self.knowledge_model.type_embeddings(random_entities).view(-1, self.config.kg_hidden_size)

                n_rel_score = self.knowledge_model._calc(
                            type_context.view(-1, self.config.kg_hidden_size), 
                            corrupt_ent_embeddings, 
                            type_rel_embeddings.view(-1, self.config.kg_hidden_size)).view(-1)

                subject_diversity_regularization = diversity_regularization(subject_ent_embeddings)
                type_diversity_regularization = diversity_regularization(type_ent_embeddings)

                type_margin_rank_loss = (self.knowledge_model.criterion(p_sub_score, n_sub_score) + \
                    self.knowledge_model.criterion(p_rel_score, n_rel_score) )/2 


                results['kg/type_margin'] = type_margin_rank_loss 
                results['kg/type_diverse'] = type_diversity_regularization
                results['kg/subject_diverse'] = subject_diversity_regularization

                total_loss += type_margin_rank_loss * self.config.kg_weight 
                    # + \
                    #  ( self.config.diversity_weight * (subject_diversity_regularization + type_diversity_regularization)) / 2


            # Softmax crossentropy
            proj_context_embeddings = self.ctx_proj(entity_output_sequence)
            
            logits = torch.mm(proj_context_embeddings.view(-1, self.config.kg_hidden_size), self.knowledge_model.ent_embeddings.weight.transpose(0, 1).detach() )
            xentropy_loss = loss_fct(logits.view(-1, self.config.ent_vocab_size ), kg_labels.view(-1) )

            results['kg/context_head_mse'] = xentropy_loss
            total_loss += xentropy_loss * self.config.kg_weight

            if self.w_kg:
                # TransE loss
                margin_loss = self.knowledge_model.calculate_loss(kg_pos_triplets, kg_neg_triplets)
                results['kg/margin_loss'] = margin_loss + self.knowledge_model.regularization(kg_pos_triplets) * self.config.kg_self_regul_weight

                if self.config.consistency_mean_loss:
                    consistency_loss, diversity_loss, type_regularization = self.knowledge_model.calculate_loss_avg(type_triplets)
                    results['kg/consistency_loss'] = consistency_loss

                    margin_loss += consistency_loss + diversity_loss + type_regularization * self.config.kg_self_regul_weight

                results['kg/all_margin_loss'] = margin_loss
                total_loss += margin_loss
        else:
            downsample_context = self.kg_downsample(entity_output_sequence)
            context_embeddings = downsample_context.view(-1, self.config.kg_hidden_size)
            proj_context_embeddings = self.ctx_proj(entity_output_sequence)

            results['proj_context_embeddings'] = proj_context_embeddings
            results['context_embeddings'] = context_embeddings

        results['loss'] = total_loss
        return results


if __name__ == '__main__':
    from transformers import PretrainedConfig, AutoTokenizer, AutoModelForMaskedLM
    from transformers.data.data_collator import DataCollatorForLanguageModeling
    from .config import FastKGBertConfig
    from .utils import init_viz_samples, mine_negative_samples
    from modules.kg.spacylink import SpacyKGTokenizer
    from modules.kg.utils import KG_DataCollatorForLanguageModeling
    from transformers import PretrainedConfig, AutoTokenizer
    import numpy as np

    template_model = 'roberta-base'
    text_tokenizer = AutoTokenizer.from_pretrained(template_model)
    #                   411094
    entity_vocab_size = 553271
    relation_vocab_size = 3886
    type_vocab_size = 97127 # total number of entity sizes
    bert_config = FastKGBertConfig(
        init_layer_num=12,
        layer_stack_mult=2,
        layer_drop_prob=0.5,
        ent_vocab_size=entity_vocab_size+2,
        rel_vocab_size=relation_vocab_size,
        type_ent_vocab_size=type_vocab_size,
        pretrained_name=template_model,
        tokenizer_base=template_model,
        vocab_size=text_tokenizer.vocab_size,
        embed_size=768,
        hidden_size=768,
        kg_hidden_size=128,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        margin_weight=1.0,
        kg_weight=1,
        total_iterations=1000000,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=514,
        position_bucket_size=64,
        type_vocab_size=2, # relation types included in rel2id
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=text_tokenizer.pad_token_id,
    )

    example_batches = torch.load('trex_example_batches.pt', map_location='cpu')
    example_batch = example_batches[0]
    negative_matrix = torch.randint(high=bert_config.ent_vocab_size-1 , size=(entity_vocab_size, 100 )).long()

    # for key in example_batch.keys():
    #     if isinstance(example_batch[key], list):
    #         example_batch[key] = [e.cpu() for e in example_batch[key] ]
    #     else:
    #         example_batch[key] = example_batch[key].cpu()

    model = EntityDisambiguation(bert_config, negative_matrix)
    # negative_matrix = mine_negative_samples(model.knowledge_model.ent_embeddings.weight, neg_size=model.neg_window_size)
    # model.negative_matrix = negative_matrix

    # model = model.cuda()
    output = model(**example_batch)
    output['loss'].backward()
    print(output)
    # data_collator = DataCollatorForLanguageModeling(tokenizer=text_tokenizer, mlm=True, mlm_probability=0.15)
    # sentences = [
    #     'Coronavirus cases in the US approach 5 million',
    # ]
    # outputs = []
    # for sent in sentences:
    #     outputs.append(text_tokenizer(sent, max_length=22, padding='max_length' ))

