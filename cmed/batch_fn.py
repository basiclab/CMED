from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Tuple, Optional
from transformers import PreTrainedTokenizer
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import random


@dataclass
class KG_DataCollatorForLanguageModeling:
    """
    Data collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    tokenizer: PreTrainedTokenizer
    mlm: bool = True
    mlm_probability: float = 0.15
    elm_probability: float = 0
    elm_all_probability: float = 0
    vocab_size: int = 10000
    ent_vocab_size: int = 386690
    tokenizer_name: str = 'bert-base-cased'

    def __call__(self, examples) -> Dict[str, torch.Tensor]:
        batch = self.data_collate(examples)

        if self.mlm:
            inputs, labels, token_mask = self.mask_tokens(batch['input_ids'], batch['kg_boolean_matrix'])
            output = {"input_ids": inputs, "labels": labels }
        else:
            output =  {"input_ids": batch, "labels": batch }

        batch['input_ids'] = output['input_ids']
        batch['labels'] = output['labels']
        return batch


    def data_collate(self, batch: Dict, ignore_keys = ['kg_boolean_matrix', 'token_type_ids', 'seq_len']):
        examples = {key: []  for key in batch[0].keys() if key not in ignore_keys }
        seq_len = torch.from_numpy(np.array([ b['seq_len'] for b in batch ]))
        boolean_matrix = []
        for idx in range(len(batch)):
            for key in examples.keys():
                examples[key].append(torch.from_numpy(np.array(batch[idx][key])).long() )
            boolean_matrix.append(torch.from_numpy(batch[idx]['kg_boolean_matrix'] ).float())

        examples['input_ids'] = self._tensorize_batch(examples['input_ids'], self.tokenizer.pad_token_id)
        examples['attention_mask'] = self._tensorize_batch(examples['attention_mask'], 0)
        examples['kg_attention_mask'] = self._tensorize_batch(examples['kg_attention_mask'], 0)

        max_seq = examples['input_ids'].shape[1]
        max_kg = examples['kg_attention_mask'].shape[1]
        examples['kg_boolean_matrix'] = self._tensorize_boolean_matrix(boolean_matrix, 
            max_seq, max_kg )

        if 'robert' in self.tokenizer_name and 'token_type_ids' in examples:
            examples.pop('token_type_ids')
        if 'token_type_ids' in examples:
            examples['token_type_ids'] = self._tensorize_batch(examples['token_type_ids'], 0)

        if 'kg_ids' in examples:
            examples['kg_ids'] = self._tensorize_batch(examples['kg_ids'], -100)
            examples['subject_rel'] = self._tensorize_batch(examples['subject_rel'], -100)
            examples['type_rel'] = self._tensorize_batch(examples['type_rel'], -100)

            examples['tail_labels'] = self._tensorize_batch(examples['tail_labels'], -100)
            examples['rel_labels'] = self._tensorize_batch(examples['rel_labels'], -100)

        examples['types'] = self._tensorize_types(examples['types'], max_kg)
        examples['subjects'] = self._tensorize_types(examples['subjects'], max_kg)


        examples['kg_inputs'], examples['kg_labels'] = self.mask_ent(examples['kg_ids'], seq_len)

        return examples

    def _tensorize_types(self, subject_matrix, max_seq):
        sub_matrixes = []
        for sub in subject_matrix:
            seq_len, sample_size = sub.shape
            if max_seq - seq_len > 0:
                sub = torch.cat( [ sub, torch.full(( max_seq-seq_len, sample_size ), -100).long() ], 0)
            sub_matrixes.append(sub)
        return torch.stack(sub_matrixes).long()

    def _tensorize_boolean_matrix(self, boolean_matrix, max_seq, max_kg):
        bool_matrixes = []
        for bool_matrix in boolean_matrix:
            kg_len, seq_len = bool_matrix.shape
            if max_seq - seq_len > 0:
                bool_matrix = torch.cat( [ bool_matrix, torch.zeros(( kg_len, max_seq-seq_len )) ], 1)
            if max_kg - kg_len > 0:
                bool_matrix = torch.cat([ bool_matrix, torch.zeros( max_kg - kg_len, max_seq ) ], 0)
            bool_matrixes.append(bool_matrix)
        return torch.stack(bool_matrixes).float()
    
    def _tensorize_batch(self, examples: List[torch.Tensor], pad_token_id=-1) -> torch.Tensor:
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            if self.tokenizer._pad_token is None:
                raise ValueError(
                    "You are attempting to pad samples but the tokenizer you are using"
                    f" ({self.tokenizer.__class__.__name__}) does not have one."
                )            
            return pad_sequence(examples, batch_first=True, padding_value=pad_token_id)
    
    
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

    def mask_tokens(self, inputs: torch.Tensor, kg_boolean_matrix: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        padding_mask = inputs.eq(self.tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

        entity_token_matrix = kg_boolean_matrix.sum(1) != 0
        probability_matrix[entity_token_matrix] += self.mlm_probability # more likely to masked

        masked_indices = torch.bernoulli(probability_matrix).bool()

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size-1, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels, masked_indices