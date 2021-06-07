from absl import flags, app


FLAGS = flags.FLAGS
# model and training
flags.DEFINE_integer('total_iterations', 1000000, help='total training iteration')
flags.DEFINE_integer('batch_size', 128, "batch size")
flags.DEFINE_float('lr', 2e-4, "Generator learning rate")
flags.DEFINE_integer('max_length', 1024, "training sentence max length")
flags.DEFINE_integer('grad_accumulation', 1, "gradient accumulation")
flags.DEFINE_integer('warmup_step', 10000, help='linearly warmup lr')

flags.DEFINE_integer('init_layers', 2, help='initial transformers layer')
flags.DEFINE_integer('layer_mult', 2, help='multiplicate transformers layer size')
flags.DEFINE_integer('layer_mult_steps', 4000, help='multiply layers for each steps')
flags.DEFINE_float('layer_dropout', 0.5, help='probability layer was dropped')
flags.DEFINE_float('connection_dropout', 0.1, help='probability layer was dropped')
flags.DEFINE_float('mlm_prob', 0.15, help='probability layer was dropped')


flags.DEFINE_integer('switch_len', 20000, help='double sequence length at step')

# model config
flags.DEFINE_integer('embed_dim', 128, "embedding dimension")
flags.DEFINE_string('tokenizer_base', 'bert-base-uncased', help='corpus path')

flags.DEFINE_integer('num_layers', 12, "number of discriminator layers")
flags.DEFINE_integer('hidden_dim', 256, "hidden dimension")
flags.DEFINE_integer('intermediate_size', 1024, "hidden dimension")
flags.DEFINE_integer('heads', 4, "number of attention heads")

flags.DEFINE_boolean('pre_norm', True, 'pre norm or post norm for attention')

# training details
flags.DEFINE_boolean('use_amp', False, 'mixed precision')

flags.DEFINE_integer('seed', 0, "random seed")
flags.DEFINE_boolean('baseline', False, "is BERT baseline")

flags.DEFINE_string('name', 'bert-small', help='session training name')
flags.DEFINE_string('model_name', 'bert-small', help='model name')
flags.DEFINE_string('corpus', '', help='corpus path')
flags.DEFINE_integer('num_gpus', 1, help='number of GPUs')
flags.DEFINE_integer('num_workers', 4, help='dataloader workers')

flags.DEFINE_string('ckpt', '', help='checkpoint name')
