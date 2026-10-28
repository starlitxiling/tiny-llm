from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import mlx.core as mx

from .embedding import Embedding  # noqa: F401 - learner checkpoint dependency
from .kv_cache import TinyKvCache
from .quantize import QuantizedWeights, dequantize_linear  # noqa: F401
from .week2_kernels import (
    FastRMSNorm,  # noqa: F401 - learner checkpoint dependency
    FastRoPE,  # noqa: F401 - learner checkpoint dependency
    scaled_dot_product_attention,  # noqa: F401 - learner checkpoint dependency
    swiglu,  # noqa: F401 - learner checkpoint dependency
)
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from .basics import linear, silu


@dataclass(frozen=True)
class Week2CheckpointFeatures:
    bounded_kv_capacity: bool = False
    quantized_weights: bool = False
    fast_rms_norm: bool = False
    fast_rope: bool = False
    fast_swiglu: bool = False
    simdgroup_matmul: bool = False
    tiled_prefill_attention: bool = False


WEEK2_CHECKPOINT_FEATURES = MappingProxyType(
    {
        "kv-cache": Week2CheckpointFeatures(),
        "capacity-cache": Week2CheckpointFeatures(bounded_kv_capacity=True),
        "quantized-matvec": Week2CheckpointFeatures(
            bounded_kv_capacity=True, quantized_weights=True
        ),
        "simd-matmul": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
        ),
        "rmsnorm": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
        ),
        "rope": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
            fast_rope=True,
        ),
        "swiglu": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
        ),
        "tiled-prefill": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
            tiled_prefill_attention=True,
        ),
        "selected": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
            tiled_prefill_attention=True,
        ),
    }
)
WEEK2_CHECKPOINTS = tuple(WEEK2_CHECKPOINT_FEATURES)


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array | QuantizedWeights,
        wk: mx.array | QuantizedWeights,
        wv: mx.array | QuantizedWeights,
        wo: mx.array | QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
        use_fast_rms_norm: bool = True,
        use_fast_rope: bool = True,
        use_tiled_prefill_attention: bool = False,
    ):
        assert num_heads % num_kv_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.rms_norm_eps = rms_norm_eps
        
        self.scale = mx.rsqrt(self.head_dim)
        self.rope = RoPE(self.head_dim, max_seq_len, theta)
    
        self.use_fast_rms_norm = use_fast_rms_norm
        self.use_fast_rope = use_fast_rope
        self.use_tiled_prefill_attention = use_tiled_prefill_attention
        

    def __call__(
        self,
        x: mx.array,
        offsets: int | list[int] | mx.array,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        Q = linear(x, self.wq).reshape(B, L, self.num_heads, self.head_dim)
        K = linear(x, self.wk).reshape(B, L, self.num_kv_heads, self.head_dim)
        V = linear(x, self.wv).reshape(B, L, self.num_kv_heads, self.head_dim)
        
        q_norm = RMSNorm(self.head_dim, self.q_norm, self.rms_norm_eps)
        k_norm = RMSNorm(self.head_dim, self.k_norm, self.rms_norm_eps)

        Q = q_norm(Q)
        K = k_norm(K)
        
        Q = self.rope(Q, offset=slice(offsets, offsets+L))
        K = self.rope(K, offset=slice(offsets, offsets+L))
        
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        key, value, _, _ = cache.update_and_fetch(K, V, L, mask)
        
        output = scaled_dot_product_attention_grouped(
            Q.astype(mx.float32), 
            key.astype(mx.float32),
            value.astype(mx.float32),
            self.scale, 
            mask,
        ).astype(x.dtype)
        
        output = output.transpose(0, 2, 1, 3)
        output = output.reshape(B, L, self.num_heads * self.head_dim)
        output = linear(output, self.wo)
        
        return output

class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array | QuantizedWeights,
        w_up: mx.array | QuantizedWeights,
        w_down: mx.array | QuantizedWeights,
        use_fast_swiglu: bool = True,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.use_fast_swiglu = use_fast_swiglu

    def __call__(self, x: mx.array) -> mx.array:
        x_gate = x @ self.w_gate.T
        x_up = x @ self.w_up.T
        
        hidden = silu(x_gate) * x_up
        output = hidden @ self.w_down.T
        return output


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array | QuantizedWeights,
        wk: mx.array | QuantizedWeights,
        wv: mx.array | QuantizedWeights,
        wo: mx.array | QuantizedWeights,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array | QuantizedWeights,
        w_up: mx.array | QuantizedWeights,
        w_down: mx.array | QuantizedWeights,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_fast_rms_norm: bool = True,
        use_fast_rope: bool = True,
        use_fast_swiglu: bool = True,
        use_tiled_prefill_attention: bool = False,
    ):
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        
        # self.use_fast_rms_norm = use_fast_rms_norm
        # self.use_fast_rope = use_fast_rope
        # self.use_fast_swiglu = use_fast_swiglu
        # self.use_decode_attention = use_decode_attention
        
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, rms_norm_eps)

        self.self_attn = Qwen3MultiHeadAttention(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            wq, wk, wv, wo,
            q_norm, k_norm,
            max_seq_len, theta, rms_norm_eps,
            use_fast_rms_norm, use_fast_rope, use_tiled_prefill_attention
        )
        
        self.post_attention_layernorm = RMSNorm(hidden_size, w_post_attention_layernorm, rms_norm_eps)
        self.mlp = Qwen3MLP(hidden_size, intermediate_size, w_gate, w_up, w_down, use_fast_swiglu)
        
        

    def __call__(
        self,
        x: mx.array,
        offset: int,
        cache: TinyKvCache,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        normed_x = self.input_layernorm(x)
        attn_out = self.self_attn(normed_x, offset, cache, mask)
        hidden = attn_out + x
        normed_hidden = self.post_attention_layernorm(hidden)
        mlp_out = self.mlp(normed_hidden)
        output = mlp_out + hidden
        
        return output


class Qwen3ModelWeek2:
    def __init__(
        self,
        mlx_model: Any,
        checkpoint: str = "selected",
        use_mlx_quantized_linear: bool = False,
        use_bounded_kv_capacity: bool | None = None,
        use_register_cached_rms_norm: bool | None = None,
        use_tiled_prefill_attention: bool | None = None,
    ):
        self.num_hidden_layers = mlx_model.args.num_hidden_layers
        self.num_attention_heads =mlx_model.args.num_attention_heads
        self.num_kv_heads = mlx_model.args.num_key_value_heads
        self.hidden_size = mlx_model.args.hidden_size
        self.head_dim = mlx_model.args.head_dim
        self.intermediate_size = mlx_model.args.intermediate_size
        self.rms_norm_eps = mlx_model.args.rms_norm_eps
        
        self.vocab_size = mlx_model.args.vocab_size
        self.tie_word_embeddings = mlx_model.args.tie_word_embeddings
        
        if not self.tie_word_embeddings:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head)
        self.max_position_embeddings = mlx_model.args.max_position_embeddings
        
        self.rope_theta = mlx_model.args.rope_theta
        
        # import pdb;pdb.set_trace()
        
        self.embedding = Embedding(
            self.vocab_size,
            self.hidden_size,
            dequantize_linear(mlx_model.model.embed_tokens),
        )
        
        self.use_decode_attention = False
        self.use_fast_swiglu = False
        self.use_fast_rms_norm = False
        self.use_fast_rope = False
        
        if checkpoint == "kv-cache":
            self.use_decode_attention = False
            self.use_fast_swiglu = False
        
        self.layers_inner = []
        for i in range(mlx_model.args.num_hidden_layers):
            layer = Qwen3TransformerBlock(
                num_attention_heads = self.num_attention_heads,
                num_kv_heads = self.num_kv_heads,
                hidden_size = self.hidden_size,
                head_dim = self.head_dim,
                intermediate_size = self.intermediate_size,
                rms_norm_eps = self.rms_norm_eps,
                wq = dequantize_linear(mlx_model.model.layers[i].self_attn.q_proj),
                wk = dequantize_linear(mlx_model.model.layers[i].self_attn.k_proj),
                wv = dequantize_linear(mlx_model.model.layers[i].self_attn.v_proj),
                wo = dequantize_linear(mlx_model.model.layers[i].self_attn.o_proj),
                q_norm = mlx_model.model.layers[i].self_attn.q_norm.weight,
                k_norm = mlx_model.model.layers[i].self_attn.k_norm.weight,
                w_gate = dequantize_linear(mlx_model.model.layers[i].mlp.gate_proj),
                w_down = dequantize_linear(mlx_model.model.layers[i].mlp.down_proj),
                w_up = dequantize_linear(mlx_model.model.layers[i].mlp.up_proj),
                w_input_layernorm = mlx_model.model.layers[i].input_layernorm.weight,
                w_post_attention_layernorm = mlx_model.model.layers[i].post_attention_layernorm.weight,
                max_seq_len = self.max_position_embeddings,
                theta = self.rope_theta,
                use_fast_rms_norm = self.use_fast_rms_norm,
                use_fast_rope = self.use_fast_rope,
                use_fast_swiglu = self.use_fast_swiglu,
                use_tiled_prefill_attention = False,

            )
            self.layers_inner.append(layer)      
            
        self.norm = RMSNorm(
            self.hidden_size,
            mlx_model.model.norm.weight,
            eps=self.rms_norm_eps,
        )

    def create_kv_cache(self, capacity: int | None = None) -> list[TinyKvCache]:
        from .kv_cache import TinyKvFullCache

        if capacity is not None:
            raise NotImplementedError("Bounded KV cache is not implemented yet")
        return [
            TinyKvFullCache()
            for _ in range(self.num_hidden_layers)
        ]

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
        cache: list[TinyKvCache],
        logits_to_keep: int | None = None,
    ) -> mx.array:
        assert len(cache) == self.num_hidden_layers
        
        for layer_idx, layer_cache in enumerate(cache):
            cache_offset = getattr(layer_cache, "offset", None)
            if cache_offset is not None:
                if cache_offset != offset:
                    raise ValueError(
                        f"layer {layer_idx} cache offset {cache_offset} "
                        f"does not match model offset {offset}"
                    )
        
        h = self.embedding(inputs)

        if inputs.shape[1] > 1:
            mask = "causal"
        else:
            mask = None
            
        for layer_idx in range(self.num_hidden_layers):
            h = self.layers_inner[layer_idx](
                h,
                offset,
                cache[layer_idx],
                mask,
            )
            
        h = self.norm(h)
        
        if self.tie_word_embeddings:
            return self.embedding.as_linear(h)
        else:
            return linear(h, self.w_lm_head)
        
