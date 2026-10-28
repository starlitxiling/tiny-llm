import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2
from typing import Callable


def _release_kv_cache(kv_cache):
    if kv_cache is None:
        return
    for layer in kv_cache:
        layer.release()


def simple_generate(
    model: Qwen3ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
    max_tokens: int = 256,
) -> None:
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
        raise ValueError("max_tokens must be a non-negative integer")
    if max_tokens == 0:
        return
    def _step(model, y):
        output_logits = model(y)
        logits = output_logits[:, -1, :]
        if sampler is None:
            next_token = mx.argmax(logits, axis=-1)
        else:
            next_token = sampler(logits - mx.logsumexp(logits, axis=-1, keepdims=True))
        return next_token
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if not tokens:
        raise ValueError("prompt must encode to at least one token")
    tokens = mx.array(tokens)
    tokens = mx.expand_dims(tokens, axis=0)
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    for _ in range(max_tokens):
        next_token = _step(model, tokens)
        mx.eval(next_token)          # tagged
        if next_token.item() == tokenizer.eos_token_id:
            break
        next_token = mx.expand_dims(next_token, axis=1)
        tokens = mx.concat([tokens, next_token], axis=1)
        detokenizer.add_token(next_token.item())
        print(detokenizer.last_segment, end="", flush=True)
    
    detokenizer.finalize()
    print(detokenizer.last_segment, end="\n", flush=True)


def simple_generate_with_kv_cache(
    model: Qwen3ModelWeek2,
    tokenizer: TokenizerWrapper,
    prompt: str,
    max_tokens: int = 256,
    use_bounded_kv_capacity: bool | None = None,
) -> str:
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
        raise ValueError("max_tokens must be a non-negative integer")
    if use_bounded_kv_capacity:
        raise NotImplementedError("Bounded KV cache is not implemented yet")
    if max_tokens == 0:
        return ""
    def _step(model, y, offset, kv_cache):
        output_logits = model(y, offset, kv_cache)
        logits = output_logits[:, -1, :]
        # if sampler is None:
        #     next_token = mx.argmax(logits, axis=-1)
        # else:
        #     next_token = sampler(logits)
        next_token = mx.argmax(logits, axis=-1)
        return next_token
    # import pdb;pdb.set_trace()
    kv_cache = model.create_kv_cache()
    offset = 0
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if not tokens:
        raise ValueError("prompt must encode to at least one token")
    tokens = mx.array(tokens)
    tokens = mx.expand_dims(tokens, axis=0)
    current_input = tokens
    detokenizer = tokenizer.detokenizer
    detokenizer.reset()
    for _ in range(max_tokens):
        next_token = _step(model, current_input, offset,kv_cache)
        mx.eval(next_token)          # tagged
        if next_token.item() == tokenizer.eos_token_id:
            break
        next_token = mx.expand_dims(next_token, axis=1)
        tokens = mx.concat([tokens, next_token], axis=1)
        detokenizer.add_token(next_token.item())
        print(detokenizer.last_segment, end="", flush=True)
        offset += current_input.shape[1]
        current_input = next_token
        
        
    
    detokenizer.finalize()
    print(detokenizer.last_segment, end="\n", flush=True)
    return detokenizer.text

def speculative_generate(
    draft_model: Qwen3ModelWeek2,
    model: Qwen3ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
    proposal_length: int = 4,
    max_tokens: int = 256,
) -> str:
    pass
