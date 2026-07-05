import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen3_week1 import Qwen3ModelWeek1
from .qwen3_week2 import Qwen3ModelWeek2
from typing import Callable


def simple_generate(
    model: Qwen3ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    def _step(model, y):
        output_logits = model(y)
        logits = output_logits[:, -1, :]
        if sampler is None:
            next_token = mx.argmax(logits, axis=-1)
        else:
            next_token = sampler(logits)
        return next_token
    # import pdb;pdb.set_trace()
    tokens = tokenizer.encode(prompt)
    tokens = mx.array(tokens)
    tokens = mx.expand_dims(tokens, axis=0)
    detokenizer = tokenizer._detokenizer
    detokenizer.reset()
    output = ""
    while True:
        next_token = _step(model, tokens)
        mx.eval(next_token)          # tagged
        if next_token.item() == tokenizer.eos_token_id:
            break
        next_token = mx.expand_dims(next_token, axis=1)
        tokens = mx.concat([tokens, next_token], axis=1)
        detokenizer.add_token(next_token.item())
        print(detokenizer.last_segment, end="", flush=True)
    
    return detokenizer.text
    
        
def simple_generate_with_kv_cache(
    model: Qwen3ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        pass


def speculative_generate(
    draft_model: Qwen3ModelWeek2,
    model: Qwen3ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
