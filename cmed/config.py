import logging

from transformers.configuration_utils import PretrainedConfig
from transformers import BertConfig


logger = logging.getLogger(__name__)


class FastBertConfig(BertConfig):
    model_type = "bert"

    def __init__(
        self,
        init_layer_num=2,
        layer_stack_mult=2,
        layer_drop_prob=0.3,
        tokenizer_base='bert-base-cased',
        vocab_size=30522,
        embed_size=128,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        total_iterations=1000000,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        position_bucket_size=64,
        type_vocab_size=2,
        pre_norm=True,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        num_labels=2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.layer_drop_prob = layer_drop_prob
        self.total_iterations = total_iterations

        self.layer_stack_mult = layer_stack_mult
        self.init_layer_num = init_layer_num
        self.pre_norm = pre_norm

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.position_bucket_size = position_bucket_size
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.tokenizer_base = tokenizer_base

    @staticmethod
    def from_pretrained(name):
        config = PretrainedConfig.from_pretrained(name)
        return config
   
    @staticmethod
    def from_file(name):
        import json
        with open(name+'.json', 'r') as f:
            params = json.load(f)
        return FastKGBertConfig(**params)

    def to_file(self, name):
        import json
        with open(name+'.json', 'w') as f:
            json.dump(self.__dict__, f)
            


class FastKGBertConfig(FastBertConfig):
    model_type = "bert"

    def __init__(
            self,
            inject_layers=[2],
            ent_vocab_size=46583,
            type_ent_vocab_size=10000,
            rel_vocab_size=12,
            kg_hidden_size=768,
            lm_weight=0.5,
            pretrained_kg_embedding=None,
            margin_weight=1.0,
            kg_hidden_layers=12,
            freeze_kg_weight=False,
            kg_weight=1.0,
            entity_max_length=128,
            attn_weight=0,
            L2=7.469e-12,
            diversity_weight=-1e-5,
            kg_self_regul_weight=0.0001,
            consistency_mean_loss=True,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.type_ent_vocab_size = type_ent_vocab_size
        self.entity_max_length = entity_max_length
        self.entity_bucket_size = 32
        self.L2 = L2
        self.lm_weight = lm_weight
        self.kg_self_regul_weight = kg_self_regul_weight
        assert diversity_weight < 0 # must be negative to increase diversity
        self.diversity_weight = diversity_weight
        self.kg_hidden_layers = kg_hidden_layers
        self.freeze_kg_weight = freeze_kg_weight
        self.kg_hidden_size = kg_hidden_size
        self.ent_vocab_size = ent_vocab_size
        self.rel_vocab_size = rel_vocab_size
        self.inject_layers = inject_layers
        self.pretrained_kg_embedding=pretrained_kg_embedding
        self.margin_weight = margin_weight
        self.kg_weight = kg_weight
        self.consistency_mean_loss = consistency_mean_loss
        self.attn_weight = attn_weight
