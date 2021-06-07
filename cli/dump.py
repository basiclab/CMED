from wikipedia2vec.dump_db import DumpDB
from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader
import argparse
import multiprocessing

arg_parser = argparse.ArgumentParser(description='Finetune module')
arg_parser.add_argument('--dump_file', type=str)
arg_parser.add_argument('--out_file', type=str)
arg_parser.add_argument('--cpu', type=int, default=multiprocessing.cpu_count())
arg_parser.add_argument('--chunk', type=int, default=100)

if __name__ == '__main__':
    args = arg_parser.parse_args()

    dump_reader = WikiDumpReader(args.dump_file)
    DumpDB.build(dump_reader, args.out_file, pool_size=args.pool_size, chunk_size=args.chunk)


    # dump_db = DumpDB(args.out_file)
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # sentence_tokenizer = SentenceTokenizer.from_name(sentence_tokenizer)

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # entity_vocab = EntityVocab(entity_vocab_file)
    # WikipediaPretrainingDataset.build(dump_db, tokenizer, sentence_tokenizer, entity_vocab, output_dir, **kwargs)
