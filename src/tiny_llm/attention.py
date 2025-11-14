import mlx.core as mx
from .basics import softmax, linear
import math


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:

    # import pdb; pdb.set_trace()
    scores = mx.matmul(query, key.swapaxes(-1, -2))
    # scores = mx.matmul(query, key.T) # is error for shape [2,1,4] .etc
    # scores = mx.matmul(query, key.swapaxes(-2, -1))  # same with swapaxes(-1, -2)
    
    if scale is None:
        d_k = query.shape[-1]
        # scale = math.sqrt(d_k)
        scale = 1.0 / math.sqrt(d_k)
    
    scores = scores * scale
    # scores = scores / scale   # can not passed f16 and f32
    
    if mask is not None: # for decoder
        if mask.dtype == mx.bool_:   # such mask: [[False, False,True], [False, True, True]]
            scores = mx.where(mask, -mx.inf, scores) # mask's True to inf, replace operation,not add
        else:  # such mask: [[ 0.0, 0.0, -inf], [ 0.0, -inf, -inf]]
            scores += mask
    # if mask is not None:
    #     scores += mask
    
    weight = mx.softmax(scores, axis=-1)
    
    outputs = mx.matmul(weight, value)
    return outputs
    # pass
    
    
# torch scaled_dot_product_attention
# def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
#         is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
#     L, S = query.size(-2), key.size(-2)
#     scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
#     attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
#     if is_causal:
#         assert attn_mask is None
#         temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
#         attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

#     if attn_mask is not None:
#         if attn_mask.dtype == torch.bool:
#             attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
#         else:
#             attn_bias = attn_mask + attn_bias

#     if enable_gqa:
#         key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
#         value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

#     attn_weight = query @ key.transpose(-2, -1) * scale_factor
#     attn_weight += attn_bias
#     attn_weight = torch.softmax(attn_weight, dim=-1)
#     attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
#     return attn_weight @ value


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int, # H * D
        num_heads: int, # H
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.head_dim = hidden_size // num_heads
        
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        # import pdb; pdb.set_trace()
        N, L, _ = query.shape
        _, S, _ = key.shape
        
        Q = mx.matmul(query, self.wq.T)
        K = mx.matmul(key, self.wk.T)
        V = mx.matmul(value, self.wv.T)
        
        Q = Q.reshape(N, L, self.num_heads, self.head_dim)
        K = K.reshape(N, S, self.num_heads, self.head_dim)
        V = V.reshape(N, S, self.num_heads, self.head_dim)
        
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # import pdb; pdb.set_trace()
        output = scaled_dot_product_attention_simple(Q, K, V, mask=mask)
        
        output = output.transpose(0, 2, 1, 3)
        
        output = output.reshape(N, L, self.hidden_size)
        
        output = mx.matmul(output, self.wo.T)
        
        return output
        # pass


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    pass