import mlx.core as mx


def softmax(x: mx.array, axis: int) -> mx.array:
    # Supplied for Day 1; a manual implementation is an optional bonus exercise.
    return mx.softmax(x, axis=axis)


def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    # output = x * (w.swapaxes(-1, -2)) + bias
    # output = mx.matmul(x, w.T) + bias
    if bias is not None:
        output = mx.matmul(x, w.swapaxes(-1, -2)) + bias
    else:
        output = mx.matmul(x, w.swapaxes(-1, -2))
    return output


def sigmoid(x: mx.array) -> mx.array:
    return mx.where(
        x >= 0,
        1 / (1 + mx.exp(-x)),
        mx.exp(x) / (1 + mx.exp(x)),
    )

def silu(x: mx.array) -> mx.array:
    x_f32 = x.astype(mx.float32)
    return (x_f32 * sigmoid(x_f32)).astype(x.dtype)
