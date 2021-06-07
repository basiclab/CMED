from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("name", None, "knowledge graph name")
flags.DEFINE_string("model_name", 'transe', "knowledge graph type")

flags.DEFINE_string("output_dir", None,
                     "knowledge graph output directory")
flags.DEFINE_integer("gpus", 1,
                     "number of gpus to use")
flags.DEFINE_float("lr", 0.01,
                     "learning rate")
flags.DEFINE_float("l2", 7.469e-12,
                     "L2 regularization weight")
flags.DEFINE_float("self_regul", 0,
                     "self regularization")
flags.DEFINE_float("div_regul", -1e-7,
                     "diversity regularization")

flags.DEFINE_float("norm", 1.0,
                     "distance norm L1 or L2")
flags.DEFINE_float("margin", 1,
                     "distance margin")

flags.DEFINE_float("type_weight", 0.1,
                     "diversity regularization")

flags.DEFINE_integer("accum_batch", 1,
                     "gradient batch accumulation")
flags.DEFINE_integer("num_workers", 8,
                     "dataloader workers")
flags.DEFINE_string("optimizer", 'sgd',
                     "optimizer type")

flags.DEFINE_integer("epochs", 1000,
                     "training epoch")
flags.DEFINE_integer("dimension", 300,
                     "dimension value")
flags.DEFINE_integer("val_batch_size", 2,
                     "validation batch size")
flags.DEFINE_integer("train_batch_size", 128,
                     "training batch size")
flags.DEFINE_string("backend", None,
                     "distributed backend None, dp, ddp")
flags.DEFINE_boolean('mean_loss', True, 'Include mean of type loss')

flags.DEFINE_boolean('eval', False, 'Evaluate only')

flags.DEFINE_string("ckpt", None,
                     "checkpoint path")

flags.mark_flag_as_required("name")
flags.mark_flag_as_required("output_dir")
