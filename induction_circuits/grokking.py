
import einops
import torch as t
import numpy as np

from typing import List, Union
from torch import Tensor
from transformer_lens import HookedTransformerConfig, HookedTransformer

from induction_circuits.utils import *

device = t.device("cuda" if t.cuda.is_available() else "cpu")

p = 113

cfg = HookedTransformerConfig(
    n_layers = 1,
    d_vocab = p+1,
    d_model = 128,
    d_mlp = 4 * 128,
    n_heads = 4,
    d_head = 128 // 4,
    n_ctx = 3,
    act_fn = "relu",
    normalization_type = None,
    device = device
)

model = HookedTransformer(cfg)



full_run_data = t.load("Grokking/large_files/grokking_addition_full_run.pth")
state_dict = full_run_data["state_dicts"][400]

model = load_in_state_dict(model, state_dict)

lines(
    lines_list=[
        full_run_data['train_losses'][::10],
        full_run_data['test_losses']
    ],
    labels=['train loss', 'test loss'],
    title='Grokking Training Curve',
    x=np.arange(5000)*10,
    xaxis='Epoch',
    yaxis='Loss',
    log_y=True
)

W_O = model.W_O[0]
W_K = model.W_K[0]
W_Q = model.W_Q[0]
W_V = model.W_V[0]
W_in = model.W_in[0]
W_out = model.W_out[0]
W_pos = model.W_pos
W_E = model.W_E[:-1]
final_pos_resid_initial = model.W_E[-1] + W_pos[2]
W_U = model.W_U[:, :-1]

shapes = [
    ('W_O  ', tuple(W_O.shape)),
    ('W_K  ', tuple(W_K.shape)),
    ('W_Q  ', tuple(W_Q.shape)),
    ('W_V  ', tuple(W_V.shape)),
    ('W_in ', tuple(W_in.shape)),
    ('W_out', tuple(W_out.shape)),
    ('W_pos', tuple(W_pos.shape)),
    ('W_U  ', tuple(W_U.shape)),
    ('W_E  ', tuple(W_E.shape)),
]