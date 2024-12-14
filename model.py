import jax
import jax.numpy as jnp
import jax.scipy.special as special
import flax
import flax.linen as nn

from typing import Any, Callable, Sequence

class PhyloBetaLoss(nn.Module):
    a_init: Callable = nn.initializers.constant(1)
    b_init: Callable = nn.initializers.constant(1)
    @nn.compact
    def __call__(self,
                 arr: jnp.ndarray) -> jnp.ndarray:
        # initialize parameter
        a, b = self.param('a', self.a_init, 1), self.param('b', self.b_init, 1)
        a, b = nn.relu(a), nn.relu(b)
        # shape information
        num_sequences, num_replicates = arr.shape
        # compute log-likelihood
        arr_sum = arr.sum(axis=0)
        log_beta_front = special.betaln(a + arr_sum, b + num_sequences - arr_sum)
        log_beta_back = special.betaln(a, b)
        log_beta = log_beta_front - log_beta_back
        return - log_beta.sum()