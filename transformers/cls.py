#%%
import math
import einops
import torch as t
import circuitsvis as cv
from IPython.display import display

from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate

from torch import Tensor
from jaxtyping import Float, Int


from model import Config, DemoTransformer, get_log_probs
#%%
device = t.device("cuda" if t.cuda.is_available() else "cpu")

cfg = Config()

reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"

reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
tokens = reference_gpt2.to_tokens(reference_text).to(device)

logits, cache = reference_gpt2.run_with_cache(tokens)
#%%
html = cv.attention.attention_patterns(
    tokens=reference_gpt2.to_str_tokens(reference_text),
    attention=cache["pattern", 0][0]
)
display(html)

# %%
html = cv.attention.attention_heads(
    tokens=reference_gpt2.to_str_tokens(reference_text),
    attention=cache["pattern", 0][0]
)
display(html)
#%%

demo_gpt2 = DemoTransformer(Config(debug=False)).to(device)
demo_gpt2.load_state_dict(reference_gpt2.state_dict(), strict=False)

demo_logits = demo_gpt2(tokens)

probs = demo_logits.softmax(dim=-1)
probs.shape
ll_tokens = logits.argmax(dim=-1)[0]
list(zip(reference_gpt2.tokenizer.batch_decode(tokens[0]), reference_gpt2.tokenizer.batch_decode(ll_tokens)))


# %%

pred_log_probs = get_log_probs(demo_logits, tokens)
print(f"Avg cross entropy loss: {-pred_log_probs.mean():.4f}")
print(f"Avg cross entropy loss for uniform distribution: {math.log(demo_gpt2.cfg.d_vocab):4f}")
print(f"Avg probability assigned to correct token: {pred_log_probs.exp().mean():4f}")

# %%
test_string = '''The Total Perspective Vortex derives its picture of the whole Universe on the principle of'''
for i in range(100):
    test_tokens = reference_gpt2.to_tokens(test_string).to(device)
    demo_logits = demo_gpt2(test_tokens)
    test_string += reference_gpt2.tokenizer.decode(demo_logits[-1, -1].argmax())

#%%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# add the EOS token as PAD token to avoid warnings
model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(torch_device)

# encode context the generation is conditioned on
model_inputs = tokenizer('I enjoy walking with my cute dog', return_tensors='pt').to(torch_device)

# generate 40 new tokens
greedy_output = model.generate(**model_inputs, max_new_tokens=40)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(greedy_output[0], skip_special_tokens=True))

# activate beam search and early_stopping

beam_outputs = model.generate(
    **model_inputs,
    max_new_tokens=40,
    num_beams=5,
    no_repeat_ngram_size=2,
    num_return_sequences=5,
    early_stopping=True
)
for i, beam in enumerate(beam_outputs):
    print(f"Output {i}:\n" + 100 * '-')
    print(tokenizer.decode(beam_outputs[i], skip_special_tokens=False))

from transformers import set_seed
set_seed(42)

sample_output = model.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    top_k=0
    temperature=0.8,)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))


# set seed to reproduce results. Feel free to change the seed though to get different results
set_seed(42)

# use temperature to decrease the sensitivity to low probability candidates
sample_output = model.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    top_k=160,
    #temperature=0.6,
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))

# set seed to reproduce results. Feel free to change the seed though to get different results
set_seed(42)

# set top_k to 50
sample_output = model.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    top_p=0.92,
    top_k=0
)

print("Output:\n" + 100 * '-')
print(tokenizer.decode(sample_output[0], skip_special_tokens=True))


# set seed to reproduce results. Feel free to change the seed though to get different results
set_seed(42)

# set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
sample_outputs = model.generate(
    **model_inputs,
    max_new_tokens=40,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    num_return_sequences=3,
)

print("Output:\n" + 100 * '-')
for i, sample_output in enumerate(sample_outputs):
  print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
