import argparse
import multiprocessing
import os
from transformers import AutoTokenizer
from wikipedia2vec.dump_db import DumpDB
from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader
from cmed.luke_entity_vocab import EntityVocab
from cmed.luke_pretrain_dataset import WikipediaPretrainingDataset

arg_parser = argparse.ArgumentParser(description='Finetune module')
arg_parser.add_argument('--dump_file', type=str)
arg_parser.add_argument('--out_file', type=str)
arg_parser.add_argument('--cpu', type=int, default=multiprocessing.cpu_count())
arg_parser.add_argument('--chunk', type=int, default=100)
arg_parser.add_argument('--entity_list', type=str, default='resources/final_entity_list.txt')
arg_parser.add_argument('--pool_size', type=int, default=20)
arg_parser.add_argument('--tokenizer_name', type=str, default='roberta-base')
arg_parser.add_argument('--sentence_tokenizer_name', type=str, default='en')

if __name__ == '__main__':
    args = arg_parser.parse_args()
    if not os.path.exists(args.out_file):
        dump_reader = WikiDumpReader(args.dump_file)
        DumpDB.build(dump_reader, args.out_file,
            pool_size=args.pool_size,
            chunk_size=args.chunk
        )
    
    dump_db = DumpDB(args.out_file)
    white_list = []
    with open(args.entity_list, 'r') as f:
        for line in f:
            white_list.append(line.replace('_', ' ').strip())

    EntityVocab.build(dump_db,
        pool_size=args.pool_size,
        vocab_size=2000000,
        white_list=white_list, 
        white_list_only=True,
        out_file=os.path.join('resources/entity_vocab.json'),
        chunk_size=2000,
        language='en'
    )    
    entity_vocab = EntityVocab(os.path.join('resources/entity_vocab.json'))
    print(len(entity_vocab))
