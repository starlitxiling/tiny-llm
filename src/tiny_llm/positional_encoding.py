import mlx.core as mx

# Rope本质上利用的是单位圆的性质，也就一个角度数值就可以表示一个形如(x_i, y_i)的二维向量


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        
        position = mx.arange(seq_len)
        # inv_freq = 1.0 / ( base ** (mx.arange(0, dims, 2) / dims))
        inv_freq = 1.0 / mx.power( base, (mx.arange(0, dims, 2) / dims))
        freqs = mx.outer(position, inv_freq)
        self.cos_freqs = mx.cos(freqs)
        self.sin_freqs = mx.sin(freqs)
           
        # pass

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        # import pdb;pdb.set_trace()

        N, L, H, D = x.shape
        # x = x.reshape(N, L, H, D // 2, 2)
        
        if offset is not None:
            cos = self.cos_freqs[offset]
            sin = self.sin_freqs[offset]
        else:
            cos = self.cos_freqs[:L]
            sin = self.sin_freqs[:L]

        if self.traditional: 
            x = x.reshape(N, L, H, D // 2, 2)
            a = x[..., 0]
            b = x[..., 1]
        else:
            a = x[..., :D // 2]
            b = x[..., D // 2:]
    
        cos = cos.reshape(1, L, 1, D // 2)
        sin = sin.reshape(1, L, 1, D // 2)
        
        new_a = a * cos - b * sin
        new_b = b * cos + a * sin
        
        if self.traditional:
            out = mx.stack([new_a, new_b], axis=-1)
        else:
            out = mx.concat([new_a, new_b], axis=-1)
        out = out.reshape(N, L, H, D)
        
        return out.astype(x.dtype)
        
        # pass
