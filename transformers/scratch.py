import einops
import torch as t

from transformer_lens import HookedTransformer
from transformer_lens.utils import gelu_new, tokenize_and_concatenate

device = t.device("cuda" if t.cuda.is_available() else "cpu")

reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)
sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n: n[1])

lengths = dict.fromkeys(range(3, 8), "")
for tok, idx in sorted_vocab[3000:]:
    if not lengths.get(len(tok), True):
        lengths[len(tok)] = tok

for length, tok in lengths.items():
    print(f"{length}: {tok}")

reference_gpt2.to_str_tokens("Ralph", False)
reference_gpt2.to_str_tokens(" Vlad")

reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text).to(device)

logits, cache = reference_gpt2.run_with_cache(tokens)
logits.shape

probs = logits.softmax(dim=-1)
probs.shape
ll_tokens = logits.argmax(dim=-1)[0]
list(zip(reference_gpt2.tokenizer.batch_decode(tokens[0]), reference_gpt2.tokenizer.batch_decode(ll_tokens)))


prefix = reference_gpt2.to_string(tokens[0])

next_token = logits[0, -1].argmax(dim=-1)
next_char = reference_gpt2.to_string(next_token)

for i in range(10):
    tokens = t.cat([tokens, next_token[None, None]], dim=-1)
    logits = reference_gpt2(tokens)
    next_token = logits[0, -1].argmax(dim=-1)
    next_char = reference_gpt2.to_string(next_token)
    print(next_char, end="")

print(reference_gpt2.to_string(tokens[0]))


# activations
for activation_name, activation in cache.items():
    if ".0." in activation_name or "blocks" not in activation_name:
        print(f"{activation_name:30} {tuple(activation.shape)}")
# parameters
for name, param in reference_gpt2.named_parameters():
    if ".0." in name or "blocks" not in name:
        print(f"{name:18} {tuple(param.shape)}")

reference_gpt2.cfg
