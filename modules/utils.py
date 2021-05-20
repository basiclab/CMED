import random
import io

import torch
import numpy as np
from torch import nn

from matplotlib import pyplot as plt
import seaborn as sns
from torch.nn.utils.rnn import pad_sequence


def negsamp_vectorized_bsearch(pos_inds, n_items, n_samp=32, items=None):
    """ Guess and check vectorized
    Assumes that we are allowed to potentially 
    return less than n_samp samples
    """
    if items is not None:
        raw_samps_idx = np.random.randint(0, len(items)-1, size=n_samp)
        raw_samps = items[raw_samps_idx]
    else:
        raw_samps = np.random.randint(0, n_items, size=n_samp)


    if len(pos_inds) > 0:
        ss = np.searchsorted(pos_inds, raw_samps)
        pos_mask = raw_samps == np.take(pos_inds, ss, mode='clip')
        neg_inds = raw_samps[~pos_mask]
        return neg_inds
    return raw_samps

def multidimensional_shifting(num_samples, sample_size, probabilities):
    # replicate probabilities as many times as `num_samples`
    replicated_probabilities = np.tile(probabilities, (num_samples, 1))
    # get random shifting numbers & scale them correctly
    random_shifts = np.random.random(replicated_probabilities.shape)
    random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
    # shift by numbers & find largest (by finding the smallest of the negative)
    shifted_probabilities = random_shifts - replicated_probabilities
    return np.argpartition(shifted_probabilities, sample_size, axis=1)[:, :sample_size]

def gen_plot(inputs, model, text_tokens):
    results = model(**inputs, 
                return_dict=True, 
                output_hidden_states=True, 
                encoder_injection_states=True)
    offset = 2
    f, axes = plt.subplots(3, figsize=(12, 8))
    pad_str = '[PAD]'
    for idx in range(3):
        text_idx = idx+offset
        str_tokens = [ t for t in text_tokens[text_idx] if t != '[PAD]']
        pad_id = 0
        if '<s>' in str_tokens:
            pad_id = 1
            str_tokens = [ t for t in text_tokens[text_idx] if t != '<pad>']

        pad_mask = inputs['input_ids'][text_idx] != pad_id
        # print(inputs['input_ids'][text_idx])
        has_ent_ids = inputs['has_ent_ids'][text_idx]
        has_ent_mask = (has_ent_ids == 1) & pad_mask
        no_ent_mask = (has_ent_ids != 1) & pad_mask
        has_ent_id_labels = has_ent_ids[pad_mask].cpu().detach().numpy()
        # print(results['other'][1][0].shape)
        injection_attention = results['other'][-2][0][text_idx].squeeze(-1)[pad_mask].cpu().detach().numpy()

        ax = axes[idx]
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        # Draw the heatmap with the mask and correct aspect ratio
        new_matrix = np.concatenate([has_ent_id_labels.reshape(-1, 1), injection_attention], 1)
        # print(new_matrix.shape)
        # print(str_tokens)
        g = sns.heatmap(new_matrix.T, cmap=cmap, vmax=1.0, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

        ax.set_xticklabels(str_tokens)
        ax.set_yticklabels([ 'label', 'word weight', 'entity weight'])
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right",
                rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=0, ha="right",
                rotation_mode="anchor")

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()

    return buf


def init_viz_samples(text_tokenizer, data_collate):
    sentences = [
        'Coronavirus cases in the US approach 5 million',
        'county seat of El Paso County, Texas, United States',
        'Chicken Run is a 2000 British stop-motion animated comedy film',
        'Trinity College Dublin is a constituent college of the University of Dublin in Ireland',
        'Texas, El Paso Washington, New York, California, New Jersey',
    ]

    outputs = []
    for idx, sent in enumerate(sentences):
        outputs.append(text_tokenizer(sent, max_length=40, padding='max_length' ))

        if idx == 0:
            outputs[-1]['has_ent_ids'][1:3] = [1,1]
            outputs[-1]['input_kgs'][1:3] = [1,1]
            outputs[-1]['has_ent_ids'] = torch.from_numpy(np.array(outputs[-1]['has_ent_ids']))
            outputs[-1]['input_kgs'] = torch.from_numpy(np.array(outputs[-1]['input_kgs']))
    text_tokens =  []
    for idx, output in enumerate(outputs):
        text_tokens.append(text_tokenizer.tokenizer.convert_ids_to_tokens(output['input_ids']))
    inputs = data_collate(outputs)
    return inputs, text_tokens


def mine_negative_samples(embeddings, neg_size=100):
    from tqdm import tqdm
    if not embeddings.is_cuda:
        embeddings = embeddings.cuda()

    print('start entity negative mining')
    negative_candidates = torch.zeros((len(embeddings), neg_size)).long()
    for idx, latent in tqdm(enumerate(embeddings), total=len(embeddings), dynamic_ncols=True):
        scores = torch.mm(embeddings, latent.view(-1, 1)).flatten()
        score_rank = torch.argsort(-scores)
        negative_candidates[idx, :] = score_rank[1:neg_size+1].cpu() # ignore myself
    del embeddings
    return negative_candidates


def print_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters: {:.4f}M'.format(total / 1e6))
    return total

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MarginLoss(nn.Module):

	def __init__(self, adv_temperature = None, margin = 6.0):
		super(MarginLoss, self).__init__()
		self.margin = nn.Parameter(torch.Tensor([margin]))
		self.margin.requires_grad = False
		if adv_temperature != None:
			self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True
		else:
			self.adv_flag = False
	
	def get_weights(self, n_score):
		return torch.softmax(-n_score * self.adv_temperature, dim = -1).detach()

	def forward(self, p_score, n_score):
		if self.adv_flag:
			return (self.get_weights(n_score) * torch.max(p_score - n_score, -self.margin)).sum(dim = -1).mean() + self.margin
		else:
			return (torch.max(p_score - n_score, -self.margin)).mean() + self.margin
			


import pytorch_lightning as pl
import os

class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="step_checkpoint",
        total_checkpoint=5,
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename
        self.checkpoints = []
        self.total_checkpoint = total_checkpoint

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            self.checkpoints.append(ckpt_path)
            trainer.save_checkpoint(ckpt_path)
        while len(self.checkpoints) > self.total_checkpoint:
            prev_ckpt_path = self.checkpoints.pop(0)
            try:
                if os.path.exists(prev_ckpt_path):
                    os.remove(prev_ckpt_path)
            except FileNotFoundError:
                break