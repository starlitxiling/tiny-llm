import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
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



def silu(x: mx.array) -> mx.array:
    pass