import argparse
import random
import torch
import argparse
import re
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
from transformers import RobertaTokenizer
from wikipedia2vec.dump_db import DumpDB
from cmed.luke_entity_vocab import EntityVocab
from cmed.constants import UNK_TOKEN

# global variables used in pool workers
_dump_db = _tokenizer = _sentence_tokenizer = _entity_vocab = _idx_entity = _max_num_tokens = _max_entity_length = None
_max_mention_length = _min_sentence_length = _include_sentences_without_entities = _include_unk_entities = None


arg_parser = argparse.ArgumentParser(description='Convert wiki dump to HDF5')
arg_parser.add_argument('--dump_db', type=str)
arg_parser.add_argument('--tokenizer', type=str)
arg_parser.add_argument('--entity_vocab', type=str)
arg_parser.add_argument('--output', type=str)



def process_page(page_title):

    if _entity_vocab.contains(page_title, _language):
        page_id = _entity_vocab.get_id(page_title, _language)
    else:
        page_id = -1

    sentences = []

    def tokenize(text: str, add_prefix_space: bool):
        text = re.sub(r"\s+", " ", text).rstrip()
        if not text:
            return []
        if isinstance(_tokenizer, RobertaTokenizer):
            return _tokenizer.tokenize(text, add_prefix_space=add_prefix_space)
        else:
            return _tokenizer.tokenize(text)

    for paragraph in _dump_db.get_paragraphs(page_title):

        paragraph_text = paragraph.text

        # First, get paragraph links.
        # Parapraph links are represented its form (link_title) and the start/end positions of strings
        # (link_start, link_end).
        paragraph_links = []
        for link in paragraph.wiki_links:
            link_title = _dump_db.resolve_redirect(link.title)
            # remove category links
            if link_title.startswith("Category:") and link.text.lower().startswith("category:"):
                paragraph_text = (
                    paragraph_text[: link.start] + " " * (link.end - link.start) + paragraph_text[link.end :]
                )
            else:
                if _entity_vocab.contains(link_title, _language):
                    paragraph_links.append((link_title, link.start, link.end))
                elif _include_unk_entities:
                    paragraph_links.append((UNK_TOKEN, link.start, link.end))

        sent_spans = _sentence_tokenizer.span_tokenize(paragraph_text.rstrip())
        for sent_start, sent_end in sent_spans:
            cur = sent_start
            sent_words = []
            sent_links = []
            # Look for links that are within the tokenized sentence.
            # If a link is found, we separate the sentences across the link and tokenize them.
            for link_title, link_start, link_end in paragraph_links:
                if not (sent_start <= link_start < sent_end and link_end <= sent_end):
                    continue
                entity_id = _entity_vocab.get_id(link_title, _language)

                text = paragraph_text[cur:link_start]
                if cur == 0 or text.startswith(" ") or paragraph_text[cur - 1] == " ":
                    sent_words += tokenize(text, True)
                else:
                    sent_words += tokenize(text, False)

                link_text = paragraph_text[link_start:link_end]

                if link_start == 0 or link_text.startswith(" ") or paragraph_text[link_start - 1] == " ":
                    link_words = tokenize(link_text, True)
                else:
                    link_words = tokenize(link_text, False)

                sent_links.append((entity_id, len(sent_words), len(sent_words) + len(link_words)))
                sent_words += link_words
                cur = link_end

            text = paragraph_text[cur:sent_end]
            if cur == 0 or text.startswith(" ") or paragraph_text[cur - 1] == " ":
                sent_words += tokenize(text, True)
            else:
                sent_words += tokenize(text, False)

            if len(sent_words) < _min_sentence_length or len(sent_words) > _max_num_tokens:
                continue
            sentences.append((sent_words, sent_links))

    ret = []
    words = []
    links = []
    for i, (sent_words, sent_links) in enumerate(sentences):
        links += [(id_, start + len(words), end + len(words)) for id_, start, end in sent_links]
        words += sent_words
        if i == len(sentences) - 1 or len(words) + len(sentences[i + 1][0]) > _max_num_tokens:
            if links or _include_sentences_without_entities:
                links = links[:_max_entity_length]
                word_ids = _tokenizer.convert_tokens_to_ids(words)
                assert _min_sentence_length <= len(word_ids) <= _max_num_tokens
                entity_ids = [id_ for id_, _, _, in links]
                assert len(entity_ids) <= _max_entity_length
                entity_position_ids = itertools.chain(
                    *[
                        (list(range(start, end)) + [-1] * (_max_mention_length - end + start))[:_max_mention_length]
                        for _, start, end in links
                    ]
                )

                input_kgs = np.array([-1]*len(word_ids))
                has_ent_ids = np.array([0]*len(word_ids))
                
                for idx, ent_pos in enumerate(entity_position_ids):        
                    ent_id = entity_ids[idx]
                    if ent_id not in _idx_entity:
                        ent_pos = entity_position_ids[idx]
                        entity_title = tokenizer.decode(word_ids[[ ent for ent in ent_pos if ent != -1]]).strip()
                    else:
                        entity_title = _idx_entity[ent_id]

                    if entity_title in _entity_vocab:
                        fastent_idx = _entity_vocab[entity_title]
                        entity_position_idx = ent_pos[ent_pos != -1]
                        input_kgs[entity_position_idx] = fastent_idx
                        has_ent_ids[entity_position_idx] = 1
                doc = {
                    'input_ids': word_ids, 
                    'input_kgs': input_kgs, 
                    'has_ent_ids': has_ent_ids, 
                    'attention_mask': np.array([1]*len(word_ids))
                }
                ret.append(doc)
            words = []
            links = []
    return ret


def init_global_args(dump_db: DumpDB,
        tokenizer,
        sentence_tokenizer,
        entity_vocab,
        idx_entity,
        max_num_tokens: int,
        max_entity_length: int,
        max_mention_length: int,
        min_sentence_length: int,
        include_sentences_without_entities: bool,
        include_unk_entities: bool,
    ):
    global _dump_db, _tokenizer, _sentence_tokenizer, _entity_vocab, _max_num_tokens, _max_entity_length
    global _max_mention_length, _min_sentence_length, _include_sentences_without_entities, _include_unk_entities
    global _language

    _dump_db = dump_db
    _idx_entity = idx_entity
    _tokenizer = tokenizer
    _sentence_tokenizer = sentence_tokenizer
    _entity_vocab = entity_vocab
    _max_num_tokens = max_num_tokens
    _max_entity_length = max_entity_length
    _max_mention_length = max_mention_length
    _min_sentence_length = min_sentence_length
    _include_sentences_without_entities = include_sentences_without_entities
    _include_unk_entities = include_unk_entities
    _language = dump_db.language

def preprocess_dump(dump_db, tokenizer, entity_vocab, max_seq_length):
    target_titles = [
        title
        for title in dump_db.titles()
        if not (":" in title and title.lower().split(":")[0] in ("image", "file", "category"))
    ]
    random.shuffle(target_titles)

    if max_num_documents is not None:
        target_titles = target_titles[:max_num_documents]

    max_num_tokens = max_seq_length - 2  # 2 for [CLS] and [SEP]

    initargs = (
        dump_db,
        tokenizer,
        sentence_tokenizer,
        entity_vocab,
        idx_entity,
        max_num_tokens,
        max_entity_length,
        max_mention_length,
        min_sentence_length,
        include_sentences_without_entities,
        include_unk_entities,
    )


    with Pool(pool_size, initializer=init_global_args, initargs=initargs) as pool:
        for ret in pool.imap(process_page, target_titles, chunksize=chunk_size):
            for sentence in ret:
                if sentence['has_ent_ids'].sum() > 0:
                    yield sentence


if __name__ == '__main__':
    from h5record import H5Dataset, Sequence
    from transformers import AutoTokenizer

    args = arg_parser.parse_args()
    output_file = args.output
    if '.h5' not in args.output:
        output_file = args.output + '.h5'


    dump_db = DumpDB(args.dump_db)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    entity_vocab = EntityVocab(args.entity_vocab)

    schema = (
        Sequence('input_ids'),
        Sequence('input_kgs'),
        Sequence('has_ent_ids'),
        Sequence('attention_mask')
    )
    generator = preprocess_dump( dump_db, tokenizer, entity_vocab)

    dataset = H5Dataset(schema, output_file, generator)
    print('Total size ', len(dataset))

