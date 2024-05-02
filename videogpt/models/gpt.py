import os
from typing import Tuple

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from videogpt.layers.norm import LayerNorm
from videogpt.layers.attention import SelfAttentionModel
from videogpt.layers.utils import shift_dim, LambdaModule

from torch_maskgit import MaskGit

MAX_SAMPLES_PER_BATCH = 32

class ImageGPT(nn.Module):
    def __init__(
        self, shape, in_features, out_features, proj_dim,
        n_head, n_layer, n_vocab, ff_mult, attn_type,
        dropout, checkpoint, attn_kwargs, cond_types):
        super().__init__()
        cond_dim = dict()
        self.cond_nets = nn.ModuleList()
        self.cond_types = cond_types
        for cond_type in cond_types:
            if cond_type.type == 'enc_attn':
                self.cond_nets.append(cond_type.net)
            else: # affine_norm
                self.cond_nets.append(nn.Identity())
            cond_dim[cond_type.type] = cond_type.out_size

        embd_dim_prev, embd_dim = in_features, out_features

        self.n_layer = n_layer
        self.shape = shape
        self.seq_len = np.product(shape)
        self.embd_dim = embd_dim
        self.n_vocab = n_vocab
        self.attn_type = attn_type
        self.cond_dim = cond_dim

        self.fc_in = nn.Linear(embd_dim_prev, embd_dim, bias=False)
        self.fc_in.weight.data.normal_(std=0.02)

        self.attn_model = SelfAttentionModel(
            shape=shape,
            n_vocab=n_vocab,
            embd_dim=embd_dim,
            proj_dim=proj_dim,
            cond_dim=cond_dim,
            n_head=n_head,
            n_layer=n_layer,
            causal=True,
            ff_mult=ff_mult,
            cond_types=cond_types,
            dropout=dropout,
            checkpoint=checkpoint,
            attn_type=attn_type,
            attn_kwargs=attn_kwargs,
        )

        self.norm = LayerNorm(embd_dim, cond_dim=cond_dim)

        self.fc_out = nn.Identity()

        self.gen_loss = nn.CrossEntropyLoss()

        self.cond_cache = None
        self._sample_idxs = self._get_sample_order(shape, attn_type, attn_kwargs)

        self.MASKGIT_T_draft = 8
        self.MASKGIT_T_revise = 8
        self.MASKGIT_M = 2
        MASKGIT_VOCAB_DIM = 256
        MASKGIT_HIDDEN_DIM = int(192/2)
        MASKGIT_SHAPE = self.shape[1:]  # self.shape is (t, h, w) = (16, 4, 4) but we do per frame masking so we remove the time dimension
        TFM_ARGS = {
            "embed_dim": MASKGIT_HIDDEN_DIM,
            "num_heads": 8,
            "num_layers": 8,
            "mlp_dim": MASKGIT_HIDDEN_DIM*4, # Can be anything
            "dropout": 0.0,
            "attention_dropout": 0.0,

            "vocab_dim": MASKGIT_VOCAB_DIM, # Required, else the code fails
            # "vocab_size": vocab_size, # DONT USE INTERNAL TF EMBEDDINGS AS TECO DOESNT EITHE
            "input_dim": embd_dim # Input dimensions for pre-encoded tokens
        }

        self.maskgit = MaskGit(MASKGIT_SHAPE,
                               n_vocab,
                               MASKGIT_VOCAB_DIM,
                               'cosine',
                               TFM_ARGS)

    def sample_order(self):
        for idx in self._sample_idxs:
            idx = tuple(idx.numpy().tolist())
            yield idx

    @staticmethod
    def _get_sample_order(shape, attn_type, attn_kwargs):
        indices = []
        for i, s in enumerate(shape):
            ind = torch.arange(s)

            view_shape = [1] * len(shape)
            view_shape[i] = s
            ind = ind.view(*view_shape, 1)

            repeat_shape = list(shape)
            repeat_shape[i] = 1
            ind = ind.repeat(*repeat_shape, 1)

            indices.append(ind)
        indices = torch.cat(indices, dim=-1)
        assert indices.shape == shape + (len(shape),)

        # raster scan order
        order = indices.flatten(end_dim=-2)
        return order

    def _clear_cache(self):
        self.cond_cache = None
        self.attn_model.clear_cache()

    def sample_mode(self):
        class ContextManager:
            def __enter__(dummy):
                self._clear_cache()

            def __exit__(dummy, *args):
                self._clear_cache()

        return ContextManager()

    def sample(self, n, codebook, cond, device, temperature, no_flatten, is_root):
        if is_root and os.environ.get('VERBOSE') == '1':
            print(f"Need {n} samples, MAX_SAMPLER_PER_BATCH = {MAX_SAMPLES_PER_BATCH}")
        samples = torch.zeros((n,) + self.shape).long().to(device)
        assert all(n == c.shape[0] for c in cond), f"cond shapes {[c.shape for c in cond]}, n, {n}"

        for i in range(0, samples.shape[0], MAX_SAMPLES_PER_BATCH):
            if is_root:
                pbar = tqdm(total=np.prod(self.shape))

            samples_subset = samples[i:i + MAX_SAMPLES_PER_BATCH]
            cond_subset = tuple([c[i:i + MAX_SAMPLES_PER_BATCH] for c in cond])

            with torch.no_grad(), self.sample_mode():
                prev_idx = None
                for j, idx in enumerate(self.sample_order()):
                    # idx must be a tuple, and not a list
                    # pytorch tensor indexing is different when using list vs tuple
                    # tuple is indexing, list is gather
                    batch_idx_slice = (slice(None, None),) + tuple([slice(i, i + 1)
                                                                    for i in idx])
                    batch_idx = (slice(None, None),) + idx

                    quantized = codebook.dictionary_lookup(samples_subset, no_flatten=True)
                    quantized = shift_dim(quantized, 1, -1)

                    if prev_idx is None:
                        s_inp = samples_subset[batch_idx_slice] # doesn't really matter what it is
                        q_inp = torch.zeros_like(quantized[batch_idx_slice])
                        s_inp. q_inp = s_inp.to(device), q_inp.to(device)
                    else:
                        s_inp, q_inp = samples_subset[prev_idx], quantized[prev_idx]

                    h = self(quantized=q_inp, encodings=s_inp, cond=cond_subset, decode_step=j,
                                  decode_idx=idx)['gen_logits']

                    logits = self.maskgit.sample(h.shape[0], self.MASKGIT_T_draft,
                                            self.MASKGIT_T_revise, self.MASKGIT_M,
                                            cond=h)

                    probs = F.softmax(logits / temperature, dim=-1)
                    if probs.shape[0] == 1:
                        probs = probs.squeeze().unsqueeze(0)
                    else:
                        probs = probs.squeeze()
                    samples_subset[batch_idx] = torch.multinomial(probs, 1).squeeze(-1)

                    prev_idx = batch_idx_slice

                    if is_root:
                        pbar.update(1)

                    if os.environ.get('DEBUG') == '1':
                        break

                if is_root:
                    pbar.close()

                samples[i:i + MAX_SAMPLES_PER_BATCH] = samples_subset

                if os.environ.get('DEBUG') == '1':
                    break

        assert samples.shape[0] == n

        encodings = samples
        quantized = codebook.dictionary_lookup(encodings, no_flatten=no_flatten)

        return quantized, encodings

    def _core(self, embeddings, cond, decode_step, decode_idx):
        return_dict = dict()
        assert len(self.cond_types) == len(cond) == len(self.cond_nets), (len(self.cond_types), len(cond), len(self.cond_nets))

        if decode_step is None or self.cond_cache is None:
            cond_map = {}
            for cond_type, cond_net, cond in zip(self.cond_types, self.cond_nets, cond):
                cond_out = cond_net(cond_type.preprocess_op(cond))
                if isinstance(cond_out, dict):
                    cond_out = cond_out['features']
                cond_map[cond_type.type] = cond_out

            if decode_step is not None:
                self.cond_cache = cond_map
        else:
            assert not self.training
            cond_map = self.cond_cache

        h = self.fc_in(embeddings)

        inps = h, cond_map, decode_step, decode_idx
        h = self.attn_model(*inps)

        h = self.norm(h, cond_map)
        gen_logits = self.fc_out(h)

        return_dict.update(gen_logits=gen_logits)
        return return_dict

    def forward_maskgit(self, targets, h):
        # h is now of shape: (batch, t, h, w, c)
        dims = h.shape

        # Combine the batch and time dimensions
        h = h.view(-1, *h.shape[2:])
        maskgit_targets = targets.view(-1, *targets.shape[2:])

        logits, labels, mask = self.maskgit(maskgit_targets, h)

        # Split the batch and time dimensions again
        logits = logits.view(dims[0], dims[1], *logits.shape[1:])
        labels = labels.view(dims[0], dims[1], *labels.shape[1:])
        mask = mask.view(dims[0], dims[1], *mask.shape[1:])

        return logits, labels, mask

    def forward(
            self,
            encodings: torch.Tensor,
            quantized: torch.Tensor,
            cond: Tuple[torch.Tensor, ...],
            decode_step=None,
            decode_idx=None,
    ):
        if self.training:
            assert decode_step is None and decode_idx is None  # FIXME: not sure

        return_dict = dict()

        """ Compute generative logits """

        return_dict.update(self._core(embeddings=quantized, cond=cond,
                                      decode_step=decode_step, decode_idx=decode_idx))

        """ Compute generative loss """

        logits, _, mask = self.forward_maskgit(encodings, return_dict['gen_logits'])

        loss = F.cross_entropy(shift_dim(logits, -1, 1), encodings, reduction='none')

        loss = (loss * mask).sum() / mask.sum()
        gen_loss = loss * np.prod(self.shape)

        return_dict.update(loss=gen_loss)

        return return_dict
