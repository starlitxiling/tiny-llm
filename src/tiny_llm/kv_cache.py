from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import mlx.core as mx

if TYPE_CHECKING:
    from .paged_kv_cache import PagedKvMetadata


class TinyKvCache(ABC):
    @abstractmethod
    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        """
        Update the key-value cache and fetch the updated key-value cache.

        Args:
            key: The key to update the cache with.
            value: The value to update the cache with.
            mask_length: The length of the mask (only used in batching mode)
            mask: The mask to use (only used in batching mode)

        Returns:
            The updated keys, updated values, sequence length, and mask. On
            On Week 2 Day 1, the mask is passed through unchanged. Week 3 Day 1
            uses the sequence length and mask to construct a dense batch.
        """
        L_new = key.shape[2]
        
        if self.key_values is None:
            self.key_values = (key, value)
        else:
            cached_key, cached_value = self.key_values
            self.key_values = (
                mx.concat(cached_key, key, axis=2), 
                mx.concat(cached_value, value, axis=2)    
            )
        self.offset += L_new
        key, value = self.key_values
        
        return key, value, self.offset, mask

    def release(self):
        pass

    def materialize(self):
        """Evaluate owned K/V storage without changing its logical layout."""
        pass

    def update_and_fetch_paged(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> "PagedKvMetadata":
        pass

    def rewind(self, n: int):
        pass


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int | None = None):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        pass

    def update_and_fetch_paged(
        self,
        keys: mx.array,
        values: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> "PagedKvMetadata":
        pass

    def add_request(self, prefilled: TinyKvCache, id: int):
        pass

    def remove_request(self, id: int):
        pass


class TinyKvFullCache(TinyKvCache):
    def __init__(self, capacity: int | None = None):
        if capacity is not None and (
            not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 0
        ):
            raise ValueError("capacity must be a non-negative integer or None")
        self.key_values = None
        self.offset = 0
        self.capacity = capacity
        self.logical_copy_bytes = 0
        self.physical_growth_copy_bytes = 0
        self.slice_write_bytes = 0
        self.growth_copy_bytes = 0

    @property
    def uses_capacity(self) -> bool:
        return self.capacity is not None

    def _logical_key_values(self) -> tuple[mx.array, mx.array]:
        """Return only initialized tokens, never the unused physical tail."""
        key, value = self.key_values
        return (
            key[:,:,:self.offset,:],
            value[:,:,:self.offset,:],
        )

    def update_and_fetch(
        self,
        key: mx.array,
        value: mx.array,
        mask_length: int | None = None,
        mask: mx.array | str | None = None,
    ) -> tuple[mx.array, mx.array, int, Optional[mx.array]]:
        if key.shape != value.shape:
            raise ValueError("Key and Value shape must match")
        
        B, H, L_new, D = key.shape

        if self.uses_capacity:
            end = self.offset + L_new

            if end > self.capacity:
                raise ValueError(
                    f"KV cache capacity {self.capacity} exceeded "
                    f"by append ending at {end}"
                )
            expected_shape = (B, H, self.capacity, D)
            
            if self.key_values is None:
                keys = mx.zeros(expected_shape, dtype=key.dtype)
                values = mx.zeros(expected_shape, dtype=value.dtype)
            else:
                keys, values = self.key_values
                if keys.shape != expected_shape or values.shape != expected_shape:
                    raise ValueError("Cache shape does not match the incoming tensors")
                if keys.dtype != key.dtype or values.dtype != value.dtype:
                    raise ValueError("Cache dtype does not match the incoming tensors")
            
            if L_new > 0:
                start = mx.array([self.offset], dtype=mx.int32)
                keys = mx.slice_update(
                    keys, key, start_indices=start, axes=(2,)
                )
                values = mx.slice_update(
                    values, value, start_indices=start, axes=(2,)
                )
            
            self.key_values = (keys, values)
            self.offset = end
            self.slice_write_bytes += key.nbytes + value.nbytes
            
            logical_keys, logical_values = self._logical_key_values()
            return logical_keys, logical_values, self.offset, mask
            
            
        else:
            if self.key_values is None:
                self.key_values = (key, value)
            else:
                cached_key, cached_value = self.key_values
                
                self.key_values = (
                    mx.concat([cached_key, key], axis=2),
                    mx.concat([cached_value, value], axis=2),
                )
                
            self.offset += L_new
            key, value = self.key_values
            return key, value, self.offset, mask
        

        
    def materialize(self):
        if self.key_values is not None:
            mx.eval(*self.key_values)

    def reset(self):
        """Reset logical length while retaining bounded physical storage."""
        self.offset = 0
        if not self.uses_capacity:
            self.key_values = None

    def rewind(self, n: int):
        if (
            not isinstance(n, int)
            or isinstance(n, bool)
            or not 0 <= n <= self.offset
        ):
            raise ValueError(
                "Rewind length must be between zero and the cache length"
            )

        self.offset -= n
        
        if not self.uses_capacity and self.key_values is not None:
            self.key_values = self._logical_key_values()