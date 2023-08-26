import einops
import torch as t
import numpy as np
import torch.nn.functional as F

import plotly.graph_objects as go
import plotly.express as px

from transformer_lens import utils

def load_in_state_dict(model, state_dict):
    '''
    Helper function to load in state dict, and do appropriate weight transpositions etc.
    '''
    state_dict_new = model.state_dict()

    # Transpose
    for k, v in state_dict.items():
        if "W_" in k:
            if "W_U" in k or "W_pos" in k:
                state_dict_new[k] = v
            elif "W_O" in k:
                state_dict_new[k] = einops.rearrange(v, "d_model (n_heads d_head) -> n_heads d_head d_model", d_head=model.cfg.d_head)
            else:
                state_dict_new[k] = v.transpose(-1, -2)
        elif "b_" in k:
            state_dict_new[k] = v

    # Make sure biases are zero
    for k, v in state_dict_new.items():
        if "b_" in k and "mlp" not in k:
            state_dict_new[k] = t.zeros_like(v)
    state_dict_new["blocks.0.attn.IGNORE"] = t.full_like(state_dict_new["blocks.0.attn.IGNORE"], -1e10)

    model.load_state_dict(state_dict_new)
    return model


def lines(lines_list, x=None, mode='lines', labels=None, xaxis='', yaxis='', title = '', log_y=False, hover=None, **kwargs):
    # Helper function to plot multiple lines
    if type(lines_list)==t.Tensor:
        lines_list = [lines_list[i] for i in range(lines_list.shape[0])]
    if x is None:
        x=np.arange(len(lines_list[0]))
    fig = go.Figure(layout={'title':title})
    fig.update_xaxes(title=xaxis)
    fig.update_yaxes(title=yaxis)
    for c, line in enumerate(lines_list):
        if type(line)==t.Tensor:
            line = utils.to_numpy(line)
        if labels is not None:
            label = labels[c]
        else:
            label = c
        fig.add_trace(go.Scatter(x=x, y=line, mode=mode, name=label, hovertext=hover, **kwargs))
    if log_y:
        fig.update_layout(yaxis_type="log")
    fig.show()

def cross_entropy_high_precision(logits, labels):
    # Shapes: batch x vocab, batch
    # Cast logits to float64 because log_softmax has a float32 underflow on overly
    # confident data and can only return multiples of 1.2e-7 (the smallest float x
    # such that 1+x is different from 1 in float32). This leads to loss spikes
    # and dodgy gradients
    logprobs = F.log_softmax(logits.to(t.float64), dim=-1)
    prediction_logprobs = t.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -t.mean(prediction_logprobs)
    return loss
