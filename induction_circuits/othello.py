
from pathlib import Path
import torch as t
import torch.nn.functional as F
import einops
import torch.nn as nn

from jaxtyping import Float, Int
from typing import Tuple, Union, List
from torch import Tensor

from transformer_lens import utils, HookedTransformerConfig, HookedTransformer

device = t.device("cuda" if t.cuda.is_available() else "cpu")

cfg = HookedTransformerConfig(
    n_layers = 8,
    d_model = 512,
    d_head = 64,
    n_heads = 8,
    d_mlp = 2048,
    d_vocab = 61,
    n_ctx = 59,
    act_fn="gelu",
    normalization_type="LNPre",
    device=device,
)
model = HookedTransformer(cfg)
sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth")
model.load_state_dict(sd)

sample_input = t.tensor([[
    20, 19, 18, 10,  2,  1, 27,  3, 41, 42, 34, 12,  4, 40, 11, 29, 43, 13, 48, 56,
    33, 39, 22, 44, 24,  5, 46,  6, 32, 36, 51, 58, 52, 60, 21, 53, 26, 31, 37,  9,
    25, 38, 23, 50, 45, 17, 47, 28, 35, 30, 54, 16, 59, 49, 57, 14, 15, 55, 7
]]).to(device)

sample_output = t.tensor([[
    21, 41, 40, 34, 40, 41,  3, 11, 21, 43, 40, 21, 28, 50, 33, 50, 33,  5, 33,  5,
    52, 46, 14, 46, 14, 47, 38, 57, 36, 50, 38, 15, 28, 26, 28, 59, 50, 28, 14, 28,
    28, 28, 28, 45, 28, 35, 15, 14, 30, 59, 49, 59, 15, 15, 14, 15,  8,  7,  8
]]).to(device)

assert (model(sample_input).argmax(dim=-1) == sample_output.to(device)).all()

#! git subtree add https://github.com/likenneth/othello_world master --prefix othello_world --squash

import os, sys
if not __file__:
    __file__ = os.getcwd()
ROOT = Path(__file__)
OTHELLO_ROOT = (ROOT / "diff_code" / "othello_world" / "mechanistic_interpretability").resolve()
sys.path.append(str(OTHELLO_ROOT))

from mech_interp_othello_utils import plot_board, plot_single_board, plot_board_log_probs, to_string, to_int, int_to_label, string_to_label, OthelloBoardState # type: ignore


