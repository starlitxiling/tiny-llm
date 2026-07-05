import mlx.core as mx
import copy


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: mx.array):
        if temp == 0:
            return mx.argmax(logprobs, axis=-1)
        logprobs = copy.copy(logprobs)

        if top_k is not None and top_k > 0:
            # import pdb;pdb.set_trace()
            k = min(top_k, logprobs.shape[-1])
            if k < logprobs.shape[-1]:
                top_indices = mx.argpartition(-logprobs, kth=k - 1, axis=-1)[
                    :, :k
                ]
                top_values = mx.take_along_axis(logprobs, top_indices, axis=-1)
                cutoff = mx.min(top_values, axis=-1, keepdims=True)
                logprobs = mx.where(logprobs < cutoff, -mx.inf, logprobs)
        if top_p is not None and top_p > 0:
            # import pdb;pdb.set_trace()
            idx = mx.argsort(logprobs)[:, ::-1]
            sorted_probs = mx.take_along_axis(
                logprobs,
                idx,
                axis=-1,
            )
            cum = mx.cumsum(mx.exp(sorted_probs), axis=-1)
            mask_elements = cum < top_p
            mask_elements[..., 0] = True
            logprobs[:, idx] = mx.where(mask_elements, sorted_probs, -mx.inf)

        logprobs = logprobs / temp
        return mx.random.categorical(logprobs, axis=-1)

    return sample
