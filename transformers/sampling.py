from dataclasses import dataclass
import einops
from rich.table import Table
from rich import print as rprint

from typing import List, Optional, Tuple
import numpy as np
import torch as t
from torch import Tensor
from jaxtyping import Float, Int
from tqdm import tqdm
from transformer_lens import HookedTransformer

from model import Config, DemoTransformer
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast

device = t.device("cuda" if t.cuda.is_available() else "cpu")

class TransformerSampler:

    def __init__(self, model: DemoTransformer, tokenizer: GPT2TokenizerFast):
        self.model = model
        self.cfg = model.cfg
        self.tokenizer = tokenizer

    @t.inference_mode()
    def sample(self, prompt: str, max_tokens_generated=100, verbose=False, **kwargs):
        '''
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an
        end-of-sequence token.

        kwargs are passed to sample_next_token, to give detailed instructions on how
        new tokens are chosen.
        '''
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)[0]

        for i in range(max_tokens_generated):
            # Get new logits (make sure we don't pass in more tokens than the model's context length)
            logits = self.model(input_ids[None, -self.cfg.n_ctx:])
            # We only take logits for the last token, because this is what we're sampling
            logits = logits[0, -1]
            # Get next token (as a tensor of size (1, 1) so we can concat it to input_ids)
            next_token = t.tensor([TransformerSampler.sample_next_token(input_ids, logits, **kwargs)], device=device)
            # Create new input ids string, with shape (1, old_seq_len + 1)
            input_ids = t.cat([input_ids, next_token], dim=-1)
            # Print out results, if required
            if verbose:
                print(self.tokenizer.decode(input_ids), end="\r")
            # If our new token was the end-of-text token, stop
            if next_token == getattr(self.tokenizer, "eos_token_id", None):
                break

        return self.tokenizer.decode(input_ids)



    @t.inference_mode()
    def beam_search(
        self,
        prompt: str,
        num_return_sequences: int,
        num_beams: int,
        max_new_tokens: int,
        no_repeat_ngram_size: int = 0,
        verbose=False
    ) -> List[Tuple[float, t.Tensor]]:
        '''
        Returns a string of autoregressively generated text, starting from the prompt.

        Sampling terminates at max_tokens_generated, or when the model generates an
        end-of-sequence token.

        kwargs are passed to sample_next_token, to give detailed instructions on how
        new tokens are chosen.
        '''
        pass


    @staticmethod
    def sample_next_token(
        input_ids: Int[Tensor, "seq_len"],
        logits: Float[Tensor, "seq_len d_vocab"],
        temperature=1.0,
        top_k=0,
        top_p=0.0,
        frequency_penalty=0.0,
        seed=None
    ):
        assert input_ids.ndim == 1, "input_ids should be a 1D sequence of token ids"
        assert temperature >= 0, "Temperature should be non-negative"
        assert 0 <= top_p <= 1.0, "Top-p must be a probability"
        assert 0 <= top_k, "Top-k must be non-negative"
        assert not (top_p != 0 and top_k != 0), "At most one of top-p and top-k supported"

        # Set random seeds for reproducibility
        if seed is not None:
            t.manual_seed(seed)
            np.random.seed(seed)

        # Apply all the specialized sampling methods
        if temperature == 0:
            return TransformerSampler.greedy_search(logits)
        elif temperature != 1.0:
            logits = TransformerSampler.apply_temperature(logits, temperature)
        if frequency_penalty != 0.0:
            logits = TransformerSampler.apply_frequency_penalty(input_ids, logits, frequency_penalty)
        if top_k > 0:
            return TransformerSampler.sample_top_k(logits, top_k)
        if top_p > 0.0:
            return TransformerSampler.sample_top_p(logits, top_p)
        return TransformerSampler.sample_basic(logits)


    @staticmethod
    def greedy_search(logits: Float[Tensor, "d_vocab"]) -> int:
        '''
        Returns the most likely token (as an int).
        '''
        out = logits.argmax().item()
        return out


    @staticmethod
    def apply_temperature(logits: Float[Tensor, "d_vocab"], temperature: float) -> Float[Tensor, "d_vocab"]:
        '''
        Applies temperature scaling to the logits.
        '''
        return logits / temperature



    @staticmethod
    def apply_frequency_penalty(input_ids: Int[Tensor, "seq_len"], logits: Float[Tensor, "d_vocab"], freq_penalty: float) -> Float[Tensor, "d_vocab"]:
        '''
        Applies a frequency penalty to the logits.
        '''
        d_vocab = logits.size(0)
        id_freqs = t.bincount(input_ids, minlength=d_vocab)
        return logits - freq_penalty * id_freqs



    @staticmethod
    def sample_basic(logits: Float[Tensor, "d_vocab"]) -> int:
        '''
        Samples from the distribution defined by the logits.
        '''
        sampled_token = t.distributions.categorical.Categorical(logits=logits).sample()
        return sampled_token.item()



    @staticmethod
    def sample_top_k(logits: Float[Tensor, "d_vocab"], k: int) -> int:
        '''
        Samples from the top k most likely tokens.
        '''
        top_k_logits, top_k_token_ids = logits.topk(k)
        # Get sampled token (which is an index corresponding to the list of top-k tokens)
        sampled_token_idx = t.distributions.categorical.Categorical(logits=top_k_logits).sample()
        # Get the actual token id, as an int
        return top_k_token_ids[sampled_token_idx].item()



    @staticmethod
    def sample_top_p(logits: Float[Tensor, "d_vocab"], top_p: float, min_tokens_to_keep: int = 1) -> int:
        '''
        Samples from the most likely tokens which make up at least p cumulative probability.
        '''
        # Sort logits, and get cumulative probabilities
        logits_sorted, indices = logits.sort(descending=True, stable=True)
        cumul_probs = logits_sorted.softmax(-1).cumsum(-1)
        # Choose which tokens to keep, in the set we sample from
        n_keep = t.searchsorted(cumul_probs, top_p, side="left").item() + 1
        n_keep = max(n_keep, min_tokens_to_keep)
        keep_idx = indices[:n_keep]
        keep_logits = logits[keep_idx]
        # Perform the sampling
        sample = t.distributions.categorical.Categorical(logits=keep_logits).sample()
        return keep_idx[sample].item()

@dataclass
class Beams:
    '''Class to store beams during beam search.'''
    model: DemoTransformer
    tokenizer: GPT2TokenizerFast
    logprob_sums: Float[Tensor, "batch"]
    tokens: Int[Tensor, "batch seq"]

    def new_beams(self, logprob_sums, tokens) -> "Beams":
        '''Creates a new Beams object with the same model and tokenizer.'''
        return Beams(self.model, self.tokenizer, logprob_sums, tokens)

    def __getitem__(self, idx) -> "Beams":
        '''Allows you to take a slice of the beams object along the batch dimension.'''
        return self.new_beams(self.logprob_sums[idx], self.tokens[idx])

    @property
    def logprobs_and_completions(self) -> List[Tuple[float, str]]:
        '''Returns self as a list of logprob sums and completions (useful for getting final output).'''
        return [
            (logprob_sum.item(), self.tokenizer.decode(tokens))
            for (logprob_sum, tokens) in zip(self.logprob_sums, self.tokens)
        ]


    def generate(self, toks_per_beam: int, no_repeat_ngram_size: Optional[int] = None) -> "Beams":
        '''
        Starting from the current set of beams (which has length `num_beams`), returns a new
        set of `num_beams * toks_per_beam`, containing the best `toks_per_beam` continuations for each
        of the original beams.

        Optional argument `no_repeat_ngram_size` means your model won't generate any sequences with
        a repeating n-gram of this length.
        '''
        # Get the output logprobs for the next token (for every sequence in current beams)
        logprobs: Tensor = self.model(self.tokens)[:, -1, :].log_softmax(-1)

        # Get the top `toks_per_beam` tokens for each sequence
        # topk_logprobs, topk_tokenIDs = logprobs.topk(k=toks_per_beam)
        topk_logprobs, topk_tokenIDs = self.get_topk_non_repeating(logprobs, no_repeat_ngram_size, k=toks_per_beam)

        # Get all of the new possible beams, via einops operations
        #   Here, we're effectively flattening out the batch dimension and k dimension, to give us tensors
        #   with every possible combination of (original sequence, new token) pairs.)
        new_logprob_sums = sum([
            einops.repeat(self.logprob_sums, "batch -> (batch k)", k=toks_per_beam),
            einops.rearrange(topk_logprobs, "batch k -> (batch k)")
        ])
        new_tokens = t.concat([
            einops.repeat(self.tokens, "batch seq -> (batch k) seq", k=toks_per_beam),
            einops.rearrange(topk_tokenIDs, "batch k -> (batch k) 1")
        ], dim=-1)
        return self.new_beams(new_logprob_sums, new_tokens)


    def filter(self, num_beams: int) -> Tuple["Beams", "Beams"]:
        '''
        Returns:
            best_beams: Beams
                filtered version of self, containing all best `num_beams` which are also not terminated.

            early_terminations: Beams
                filtered version of self, containing all best `num_beams` which are also terminated.
                i.e. the sum of lengths of these two should equal `num_beams`.
        '''
        # Get the indices of top `num_beams` beams
        top_beam_indices = self.logprob_sums.topk(k=num_beams, dim=0).indices.tolist()
        # Get the indices of terminated sequences
        new_tokens = self.tokens[:, -1]
        terminated_indices = t.nonzero(new_tokens == self.tokenizer.eos_token_id)

        # Get the indices of the `num_beams` best sequences (some terminated, some not terminated)
        best_continuing = [i for i in top_beam_indices if i not in terminated_indices]
        best_terminated = [i for i in top_beam_indices if i in terminated_indices]

        # Return the beam objects from these indices
        return self[best_continuing], self[best_terminated]


    def print(self, title="Best completions", max_print_chars=80) -> None:
        '''
        Prints out a set of sequences with their corresponding logitsums.
        '''
        if len(self.tokens) == 0:
            return
        table = Table("logitsum", "completion", title=title)
        for logprob_sum, tokens in zip(self.logprob_sums, self.tokens):
            text = self.tokenizer.decode(tokens)
            if len(repr(text)) > max_print_chars:
                text = text[:int(0.3 * max_print_chars)] + " ... " + text[-int(0.7 * max_print_chars):]
            table.add_row(f"{logprob_sum:>8.3f}", repr(text))
        rprint(table)


    def get_topk_non_repeating(
        self,
        logprobs: Float[Tensor, "batch d_vocab"],
        no_repeat_ngram_size: Optional[int],
        k: int,
    ) -> Tuple[Float[Tensor, "k"], Int[Tensor, "k"]]:
        '''
        logprobs:
            tensor of the log-probs for the next token
        no_repeat_ngram_size:
            size of ngram to avoid repeating
        k:
            number of top logits to return, for each beam in our collection

        Returns:
            equivalent to the output of `logprobs.topk(dim=-1)`, but makes sure
            that no returned tokens would produce an ngram of size  `no_repeat_ngram_size`
            which has already appeared in `self.tokens`.
        '''
        batch, seq_len = self.tokens.shape
        neg_inf = t.tensor(-1.0e4).to(device)

        # If completion isn't long enough for a repetition, or we have no restructions, just return topk
        if (no_repeat_ngram_size is not None) and (seq_len > no_repeat_ngram_size-1):
            # Otherwise, we need to check for ngram repetitions
            # First, get the most recent `no_repeat_ngram_size-1` tokens
            last_ngram_prefix = self.tokens[:, seq_len - (no_repeat_ngram_size-1):]
            # Next, find all the tokens we're not allowed to generate (by going iterating through past ngrams and seeing if those ngram prefixes match the last one)
            for i in range(seq_len - (no_repeat_ngram_size-1)):
                ngrams = self.tokens[:, i:i+no_repeat_ngram_size] # (batch, ngram)
                ngrams_are_repeated = (ngrams[:, :-1] == last_ngram_prefix).all(-1) # (batch,)
                ngram_end_tokens = ngrams[:, [-1]] # (batch, 1)
                # Fill logprobs with neginf wherever the ngrams are repeated
                logprobs[range(batch), ngram_end_tokens] = t.where(
                    ngrams_are_repeated,
                    neg_inf,
                    logprobs[range(batch), ngram_end_tokens],
            )

        # Finally, get our actual tokens
        return logprobs.topk(k=k, dim=-1)


@t.inference_mode()
def beam_search(
    self: TransformerSampler,
    prompt: str,
    num_return_sequences: int,
    num_beams: int,
    max_new_tokens: int,
    no_repeat_ngram_size: Optional[int] = None,
    verbose=False
) -> List[Tuple[float, Tensor]]:
    '''
    Implements a beam search, by repeatedly performing the `generate` and `filter` steps (starting
    from the initial prompt) until either of the two stopping criteria are met:

        (1) we've generated `max_new_tokens` tokens, or
        (2) we've generated `num_returns_sequences` terminating sequences.

    To modularize this function, most of the actual complexity is in the Beams class,
    in the `generate` and `filter` methods.
    '''

    assert num_return_sequences <= num_beams
    self.model.eval()

    tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(device)

    # List for final beams to return (and early terminations)
    final_logprobs_and_completions: List[Tuple[float, str]] = []
    # Keep track of all best beams after each step
    best_beams = Beams(self.model, self.tokenizer, t.tensor([0.0]).to(device), tokens)

    for n in tqdm(range(max_new_tokens)):

        # Generation step
        best_beams = best_beams.generate(toks_per_beam=num_beams, no_repeat_ngram_size=no_repeat_ngram_size)

        # Filtering step
        best_beams, best_beams_terminated = best_beams.filter(num_beams=num_beams)
        final_logprobs_and_completions.extend(best_beams_terminated.logprobs_and_completions)

        # Print output
        if verbose:
            best_beams.print()

        # Check stopping condition
        if len(final_logprobs_and_completions) >= num_return_sequences:
            return final_logprobs_and_completions[:num_return_sequences]

    final_logprobs_and_completions.extend(best_beams.logprobs_and_completions)
    final_logprobs_and_completions = final_logprobs_and_completions[:num_return_sequences]
    return final_logprobs_and_completions



if __name__ == "__main__":
    from utils import generator_audio
    reference_gpt2 = HookedTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)

    model_cfg = Config()
    model = DemoTransformer(model_cfg).to(device)
    model.load_state_dict(reference_gpt2.state_dict(), strict=False)

    tokenizer = reference_gpt2.tokenizer

    sampler = TransformerSampler(model, tokenizer)
    TransformerSampler.beam_search = beam_search

    prompt = "Jingle bells, jingle bells, jingle all the way"
    prompt = "John and Mary went to the"
    prompt = "And I was like Baby, baby, baby, oh Like, Baby, baby, baby, no Like, Baby, baby, baby, oh I thought you'd always be mine, mine"
    prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
    prompt = "Eliezer Shlomo Yudkowsky (born September 11, 1979) is an American decision and artificial intelligence (AI) theorist and writer, best known for"
    text = sampler.sample(prompt, max_tokens_generated=80,
                          temperature=0.9, top_p=0.95,
                          frequency_penalty=3.0,
                          verbose=False)
    residual = text[len(prompt):]
    print(residual)
    generator_audio(residual)


    final_logitsums_and_completions = sampler.beam_search(
        prompt=prompt,
        num_return_sequences=2,
        num_beams=4,
        max_new_tokens=60,
        no_repeat_ngram_size=2,
        verbose=False
    )

    # Print all the best output
    orig_len = len(tokenizer.encode(prompt))
    for logprob_sum, text in final_logitsums_and_completions:
        avg_logprob_as_prob = t.tensor(logprob_sum / (len(tokenizer.encode(text)) - orig_len)).exp().item()
        print("=" * 25 + f" Avg logprob (as probability) = {avg_logprob_as_prob:.3f} " + "=" * 25)
        rprint("Best output:\n\n[bold dark_orange]" + text)
