# %%
import einops
import torch as t
from transformer_lens import HookedTransformer, utils
import circuitsvis as cv
from IPython.display import display, Markdown

#%%
device = t.device("cuda" if t.cuda.is_available() else "cpu")
model = HookedTransformer.from_pretrained("gpt2-small")

#%%
model_description_text = '''## Loading Models'''

model.blocks[0].attn.W_Q.shape # nheads, d_model, d_head
model.W_Q.shape # layer, nheads, d_model, d_head
model.W_Q[0].shape
model.W_E.shape
model.W_U.shape

tokens = model.to_tokens(model_description_text)
tokens.shape

loss = model(tokens, return_type="loss")

model.to_str_tokens(model_description_text)
model.to_string(tokens)

#%%
prompt = """ You will meet an important person who will help you advance professionally."""
prompt = """
We should be careful to get out of an experience only the wisdom that is
in it - and stay there, lest we be like the cat that sits down on a hot
stove-lid.  She will never sit down on a hot stove-lid again - and that
is well; but also she will never sit down on a cold one any more.
		-- Mark Twain

"""
#%%
logits = model(prompt, return_type="logits")
prediction = logits.argmax(dim=-1).squeeze(0)[:-1]

tokens = model.to_tokens(prompt)[0][1:]
correct = (prediction==tokens)
correct.sum().item()/correct.numel()
model.to_string(prediction[correct])

list(zip(model.to_str_tokens(tokens)[1:], model.to_str_tokens(prediction)))

logits, cache = model.run_with_cache(prompt, return_type="logits", remove_batch_dim=True)
attn_l0 = cache["pattern", 0]
q = cache["q", 0]
k = cache["k", 0]
seq, nheads, headsize = q.shape
attn_score = einops.einsum(q, k, "seqQ nheads headsize, seqK nheads headsize -> nheads seqQ seqK")
mask = t.triu(t.ones(seq, seq), diagonal=1).to(device)
attn_score.masked_fill_(mask==1, -1e9)
pattern = t.softmax(attn_score / headsize**.5, dim=-1)
t.testing.assert_allclose(pattern, attn_l0)
#%%


gpt2_small: HookedTransformer = HookedTransformer.from_pretrained("gpt2-small")
gpt2_text = "Natural language processing tasks, such as question answering, machine translation, reading comprehension, and summarization, are typically approached with supervised learning on taskspecific datasets."
gpt2_tokens = gpt2_small.to_tokens(gpt2_text)
gpt2_logits, gpt2_cache = gpt2_small.run_with_cache(gpt2_tokens, remove_batch_dim=True)

attention_pattern = gpt2_cache["pattern", 0, "attn"]
gpt2_str_tokens = gpt2_small.to_str_tokens(gpt2_text)

display(cv.attention.attention_patterns(
	tokens=gpt2_str_tokens,
	attention=attention_pattern,
	attention_head_names=[f"L0H{i}" for i in range(12)],
))


# %%
