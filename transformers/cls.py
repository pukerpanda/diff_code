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


from model import Config, DemoTransformer
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
def get_log_probs(
	logits: Float[Tensor, "batch posn d_vocab"],
	tokens: Int[Tensor, "batch posn"]
) -> Float[Tensor, "batch posn-1"]:

	log_probs = logits.log_softmax(dim=-1)
	# Get logprobs the first seq_len-1 predictions (so we can compare them with the actual next tokens)
	log_probs_for_tokens = log_probs[:, :-1].gather(dim=-1, index=tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

	return log_probs_for_tokens

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

