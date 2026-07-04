import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
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
        
        
    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        
        B, L, _ = x.shape
        
        Q = linear(x, self.wq).reshape(B, L, self.num_heads, self.head_dim)
        K = linear(x, self.wk).reshape(B, L, self.num_kv_heads, self.head_dim)
        V = linear(x, self.wv).reshape(B, L, self.num_kv_heads, self.head_dim)
        
        # Q = mx.fast.rms_norm(Q, self.q_norm, eps=self.rms_norm_eps)
        # K = mx.fast.rms_norm(K, self.k_norm, eps=self.rms_norm_eps)
        q_norm = RMSNorm(self.head_dim, self.q_norm, self.rms_norm_eps)
        k_norm = RMSNorm(self.head_dim, self.k_norm, self.rms_norm_eps)
    
        Q = q_norm(Q)
        K = k_norm(K)
        
        Q = self.rope(Q, offset=slice(0, L))
        K = self.rope(K, offset=slice(0, L)) 
        
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        output = scaled_dot_product_attention_grouped(
            Q.astype(mx.float32), 
            K.astype(mx.float32),
            V.astype(mx.float32),
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
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        # import pdb;pdb.set_trace()
        # L, E = x.shape[-2:]
        # expected_shape = x.shape
        # x = x.reshape(L,E)
        # import pdb;pdb.set_trace()
        x_gate = x @ self.w_gate.T
        x_up = x @ self.w_up.T
        
        hidden = silu(x_gate) * x_up
        output = hidden @ self.w_down.T
        # return output.reshape(expected_shape)
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
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, rms_norm_eps)
        
        self.qwen3_multi_head_attention = Qwen3MultiHeadAttention(
            hidden_size, 
            num_attention_heads, 
            num_kv_heads,
            head_dim,
            wq, wk, wv, wo,
            q_norm, k_norm,
            max_seq_len, theta, rms_norm_eps
        )
        
        self.post_attention_layernorm = RMSNorm(hidden_size, w_post_attention_layernorm, rms_norm_eps)
        
        self.mlp = Qwen3MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        normed_x = self.input_layernorm(x)
        attn_out = self.qwen3_multi_head_attention(normed_x, mask)
        hidden = attn_out + x
        normed_hidden = self.post_attention_layernorm(hidden)
        mlp_out = self.mlp(normed_hidden)
        output = mlp_out + hidden
        
        return output

class Qwen3ModelWeek1:
    def __init__(self, mlx_model: Any):
        # import pdb;pdb.set_trace()
        self.num_attention_heads = mlx_model.args.num_attention_heads
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
        
        self.embedding = Embedding(
            self.vocab_size,
            self.hidden_size,
            dequantize_linear(mlx_model.model.embed_tokens),
        )
        # import pdb;pdb.set_trace()
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
            )
            self.layers_inner.append(layer)      
            
        self.norm = RMSNorm(
                self.hidden_size,
                mlx_model.model.norm.weight,
                eps=self.rms_norm_eps,
            )

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        h = self.embedding(inputs)

        mask = None
        
        if h.shape[-2] > 1:
            mask = 'causal'
        
        # h = self.layers_inner(h, mask)
        for i in range(len(self.layers_inner)):
            h = self.layers_inner[i](h, mask)
        
        h = self.norm(h)
        
        if self.tie_word_embeddings:
            return self.embedding.as_linear(h)
        else:
            return linear(h, self.w_lm_head)
        
