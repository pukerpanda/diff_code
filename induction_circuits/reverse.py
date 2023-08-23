
import json
from typing import List, Optional, Tuple, Union
import einops
import torch as t

from transformer_lens import utils, HookedTransformer, HookedTransformerConfig, ActivationCache
from jaxtyping import Float, Int
from torch import Tensor
from fundamentals.resnet import plotly_utils

from induction_circuits.brackets import SimpleTokenizer, BracketsDataset
from transformer_lens.hook_points import HookPoint
from transformer_lens.components import LayerNorm

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

VOCAB = "()"

cfg = HookedTransformerConfig(
    n_ctx=42,
    d_model=56,
    d_head=28,
    n_heads=2,
    d_mlp=56,
    n_layers=3,
    attention_dir="bidirectional", # defaults to "causal"
    act_fn="relu",
    d_vocab=len(VOCAB)+3, # plus 3 because of end and pad and start token
    d_vocab_out=2, # 2 because we're doing binary classification
    use_attn_result=True,
    device=device,
    use_hook_tokens=True
)

model = HookedTransformer(cfg).eval()

weigths = './chapter1_transformers/exercises/part4_interp_on_algorithmic_model/brackets_model_state_dict.pt'

state_dict = t.load(weigths, map_location=device)
model.load_state_dict(state_dict)

tokenizer = SimpleTokenizer(VOCAB)

tokenizer.i_to_t
tokenizer.t_to_i

def hooks_to_mask_pad_tokens(model, pad_tokens):
    def cache_pad_tokens_mask(tokens: Float[Tensor, "batch seq"], hook):
        hook.ctx["pad_tokens_mask"] = einops.rearrange(tokens == pad_tokens, "batch seq -> batch 1 1 seq")

    def cache_padding_tokens_mask(tokens: Float[Tensor, "batch seq"], hook: HookPoint) -> None:
        hook.ctx["padding_tokens_mask"] = einops.rearrange(tokens == pad_tokens, "b sK -> b 1 1 sK")

    def apply_pad_tokens_mask(attn_scores: Float[Tensor, "batch heads seqQ seqK"], hook):
        attn_scores.masked_fill_(model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"], -1e5)
        if hook.layer() == model.cfg.n_layers - 1:
            del model.hook_dict["hook_tokens"].ctx["padding_tokens_mask"]

    for name, hook in model.hook_dict.items():
        if name == "hook_tokens":
            hook.add_perma_hook(cache_padding_tokens_mask)
        elif name.endswith("attn_scores"):
            hook.add_perma_hook(apply_pad_tokens_mask)

    return model

model.reset_hooks(including_permanent=True)
model = hooks_to_mask_pad_tokens(model, tokenizer.PAD_TOKEN)

brackets_data = './chapter1_transformers/exercises/part4_interp_on_algorithmic_model/brackets_data.json'

N_SAMPLES = 5000
with open(brackets_data) as f:
    data_t = json.load(f)
len(data_t)
data = BracketsDataset(data_t[:N_SAMPLES]).to(device)
data_mini = BracketsDataset(data_t[:100]).to(device)

from fundamentals.resnet.plotly_utils import hist, line
# hist([len(s[0]) for s in data_t], nbins=data.seq_length)

examples = ["()()", "(())", "))((", "()", "((()()()()))", "(()()()(()(())()", "()(()(((())())()))"]
labels = [True, True, False, True, True, False, True]
tokens = tokenizer.tokenize(examples)

# Get output logits for the 0th sequence position (i.e. the [start] token)
logits = model(tokens)[:, 0]

prob_balanced = logits.softmax(-1)[:, 1]
print("Model confidence:\n" + "\n".join([f"{ex:18} : {prob:<8.4%} : label={int(label)}" for ex, prob, label in zip(examples, prob_balanced, labels)]))


def is_balanced(tokens: Int[Tensor, "seq"]) -> bool:
    table = t.tensor([0, 0, 0, 1, -1], device=device)
    change = table[tokens]
    alt = t.cumsum(change, -1)
    return t.logical_and((alt[...,-1] == 0), (alt.min(-1)[0] >= 0))

# tokens = tokenizer.tokenize(examples[1])
#t.all(is_balanced(tokens) == t.tensor(labels, device=device))

post_final_ln_dir = lambda model: model.W_U[:,0] - model.W_U[:,1]

pfld = post_final_ln_dir(model)

model.W_U @ logits[0].T

model.cfg
model.W_U.shape

def get_activations(
    model: HookedTransformer,
    toks: Int[Tensor, "batch seq"],
    names: Union[str, List[str]]
) -> Union[t.Tensor, ActivationCache]:
    '''
    Uses hooks to return activations from the model.

    If names is a string, returns the activations for that hook name.
    If names is a list of strings, returns a dictionary mapping hook names to tensors of activations.
    '''
    names_list = [names] if isinstance(names, str) else names
    _, cache = model.run_with_cache(
        toks,
        return_type=None,
        names_filter=lambda name: name in names_list,
    )

    return cache[names] if isinstance(names, str) else cache

def LN_hook_names(layernorm: LayerNorm) -> Tuple[str, str]:
    '''
    Returns the names of the hooks immediately before and after a given layernorm.
    e.g. LN_hook_names(model.final_ln) returns ["blocks.2.hook_resid_post", "ln_final.hook_normalized"]
    '''
    if layernorm.name == "ln_final":
        input_hook_name = utils.get_act_name("resid_post", 2)
        output_hook_name = "ln_final.hook_normalized"
    else:
        layer, ln = layernorm.name.split(".")[1:]
        input_hook_name = utils.get_act_name("resid_pre" if ln=="ln1" else "resid_mid", layer)
        output_hook_name = utils.get_act_name('normalized', layer, ln)

    return input_hook_name, output_hook_name

pre_final_ln_name, post_final_ln_name = LN_hook_names(model.ln_final)
LN_hook_names(model.blocks[1].ln1)

from sklearn.linear_model import LinearRegression

def get_ln_fit(
	model: HookedTransformer, data: BracketsDataset, layernorm: LayerNorm, seq_pos: Optional[int] = None
) -> Tuple[LinearRegression, float]:
	'''
	if seq_pos is None, find best fit aggregated over all sequence positions. Otherwise, fit only for given seq_pos.

	Returns: A tuple of a (fitted) sklearn LinearRegression object and the r^2 of the fit
	'''
	input_hook_name, output_hook_name = LN_hook_names(layernorm)

	activations_dict = get_activations(model, data.toks, [input_hook_name, output_hook_name])
	inputs = utils.to_numpy(activations_dict[input_hook_name])
	outputs = utils.to_numpy(activations_dict[output_hook_name])

	if seq_pos is None:
		inputs = einops.rearrange(inputs, "batch seq d_model -> (batch seq) d_model")
		outputs = einops.rearrange(outputs, "batch seq d_model -> (batch seq) d_model")
	else:
		inputs = inputs[:, seq_pos, :]
		outputs = outputs[:, seq_pos, :]

	final_ln_fit = LinearRegression().fit(inputs, outputs)

	r2 = final_ln_fit.score(inputs, outputs)

	return (final_ln_fit, r2)


get_ln_fit(model, data_mini, model.ln_final, seq_pos=1)
get_ln_fit(model, data, layernorm=model.blocks[1].ln1, seq_pos=None)

def get_post_final_ln_dir(model: HookedTransformer) -> Float[Tensor, "d_model"]:
	'''
	Returns the direction in which final_ln_output[0, :] should point to maximize P(unbalanced)
	'''
	return model.W_U[:, 0] - model.W_U[:, 1]

def get_pre_final_ln_dir(model: HookedTransformer, data: BracketsDataset) -> Float[Tensor, "d_model"]:
    '''
    Returns the direction in residual stream (pre ln_final, at sequence position 0) which
    most points in the direction of making an unbalanced classification.
    '''
    post_final_ln_dir = get_post_final_ln_dir(model)

    final_ln_fit = get_ln_fit(model, data, layernorm=model.ln_final, seq_pos=0)[0]
    final_ln_coefs = t.from_numpy(final_ln_fit.coef_).to(device)

    return final_ln_coefs.T @ post_final_ln_dir

res = get_pre_final_ln_dir(model, data_mini)

def get_out_by_components(model: HookedTransformer, data: BracketsDataset) -> Float[Tensor, "component batch seq_pos emb"]:
	'''
	Computes a tensor of shape [10, dataset_size, seq_pos, emb] representing the output of the model's components when run on the data.
	The first dimension is  [embeddings, head 0.0, head 0.1, mlp 0, head 1.0, head 1.1, mlp 1, head 2.0, head 2.1, mlp 2]
	'''
	embedding_hook_names = ["hook_embed", "hook_pos_embed"]
	head_hook_names = [utils.get_act_name("result", layer) for layer in range(model.cfg.n_layers)]
	mlp_hook_names = [utils.get_act_name("mlp_out", layer) for layer in range(model.cfg.n_layers)]

	all_hook_names = embedding_hook_names + head_hook_names + mlp_hook_names
	activations = get_activations(model, data.toks, all_hook_names)

	out = (activations["hook_embed"] + activations["hook_pos_embed"]).unsqueeze(0)

	for head_hook_name, mlp_hook_name in zip(head_hook_names, mlp_hook_names):
		out = t.concat([
			out,
			einops.rearrange(activations[head_hook_name], "batch seq heads emb -> heads batch seq emb"),
			activations[mlp_hook_name].unsqueeze(0)
		])

	return out

get_out_by_components(model, data_mini).shape

biases = model.b_O.sum(0)
out_by_components = get_out_by_components(model, data)
summed_terms = out_by_components.sum(dim=0) + biases

final_ln_input_name, final_ln_output_name = LN_hook_names(model.ln_final)
final_ln_input = get_activations(model, data.toks, final_ln_input_name)

out_by_components_seq0: Float[Tensor, "comp batch d_model"] = out_by_components[:, :, 0, :]
pre_final_ln_dir: Float[Tensor, "d_model"] = get_pre_final_ln_dir(model, data)
out_by_component_in_unbalanced_dir = einops.einsum(
		out_by_components_seq0,
		pre_final_ln_dir,
		"comp batch d_model, d_model -> comp batch",
	)
out_by_component_in_unbalanced_dir -= out_by_component_in_unbalanced_dir[:, data.isbal].mean(dim=1).unsqueeze(1)

from plotly.subplots import make_subplots
from transformer_lens.utils import to_numpy
import plotly.graph_objects as go

def hists_per_comp(out_by_component_in_unbalanced_dir: Float[Tensor, "component batch"], data, xaxis_range=(-1, 1)):
    '''
    Plots the contributions in the unbalanced direction, as supplied by the `out_by_component_in_unbalanced_dir` tensor.
    '''
    titles = {
        (1, 1): "embeddings",
        (2, 1): "head 0.0", (2, 2): "head 0.1", (2, 3): "mlp 0",
        (3, 1): "head 1.0", (3, 2): "head 1.1", (3, 3): "mlp 1",
        (4, 1): "head 2.0", (4, 2): "head 2.1", (4, 3): "mlp 2"
    }
    n_layers = out_by_component_in_unbalanced_dir.shape[0] // 3
    fig = make_subplots(rows=n_layers+1, cols=3)
    for ((row, col), title), in_dir in zip(titles.items(), out_by_component_in_unbalanced_dir):
        fig.add_trace(go.Histogram(x=to_numpy(in_dir[data.isbal]), name="Balanced", marker_color="blue", opacity=0.5, legendgroup = '1', showlegend=title=="embeddings"), row=row, col=col)
        fig.add_trace(go.Histogram(x=to_numpy(in_dir[~data.isbal]), name="Unbalanced", marker_color="red", opacity=0.5, legendgroup = '2', showlegend=title=="embeddings"), row=row, col=col)
        fig.update_xaxes(title_text=title, row=row, col=col, range=xaxis_range)
    fig.update_layout(width=1200, height=250*(n_layers+1), barmode="overlay", legend=dict(yanchor="top", y=0.92, xanchor="left", x=0.4), title="Histograms of component significance")
    fig.show()


hists_per_comp(
    out_by_component_in_unbalanced_dir,
    data, xaxis_range=[-10, 20]
)

