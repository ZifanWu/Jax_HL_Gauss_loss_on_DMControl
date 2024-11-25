import functools
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.module import init
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from jaxrl.networks.common import MLP, Params, PRNGKey, default_init

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class MSEPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> jnp.ndarray:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        actions = nn.Dense(self.action_dim,
                           kernel_init=default_init())(outputs)
        return nn.tanh(actions)


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True # TODO check if we need the scheduled std in official DrQ-v2
    dropout_rate: Optional[float] = None
    final_fc_init_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True
    init_mean: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim,
                         kernel_init=default_init(
                             self.final_fc_init_scale))(outputs)
        if self.init_mean is not None:
            means += self.init_mean

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=default_init(
                                    self.final_fc_init_scale))(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros,
                                  (self.action_dim, ))

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) *
                                               temperature)
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(distribution=base_dist,
                                               bijector=tfb.Tanh())
        else:
            return base_dist


class NormalTanhMixturePolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    num_components: int = 5
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        logits = nn.Dense(self.action_dim * self.num_components,
                          kernel_init=default_init())(outputs)
        means = nn.Dense(self.action_dim * self.num_components,
                         kernel_init=default_init(),
                         bias_init=nn.initializers.normal(stddev=1.0))(outputs)
        log_stds = nn.Dense(self.action_dim * self.num_components,
                            kernel_init=default_init())(outputs)

        shape = list(observations.shape[:-1]) + [-1, self.num_components]
        logits = jnp.reshape(logits, shape)
        mu = jnp.reshape(means, shape)
        log_stds = jnp.reshape(log_stds, shape)

        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        components_distribution = tfd.Normal(loc=mu,
                                             scale=jnp.exp(log_stds) *
                                             temperature)

        base_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=logits),
            components_distribution=components_distribution)

        dist = tfd.TransformedDistribution(distribution=base_dist,
                                           bijector=tfb.Tanh())

        return tfd.Independent(dist, 1)


@functools.partial(jax.jit, static_argnames=('actor_def', 'distribution'))
def _sample_actions(
        rng: PRNGKey,
        actor_def: nn.Module,
        actor_params: Params,
        observations: np.ndarray,
        temperature: float = 1.0,
        distribution: str = 'log_prob') -> Tuple[PRNGKey, jnp.ndarray]:
    if distribution == 'det':
        return rng, actor_def.apply({'params': actor_params}, observations,
                                    temperature)
    else:
        dist = actor_def.apply({'params': actor_params}, observations,
                               temperature)
        rng, key = jax.random.split(rng)
        return rng, dist.sample(seed=key)


def sample_actions(
        rng: PRNGKey,
        actor_def: nn.Module,
        actor_params: Params,
        observations: np.ndarray,
        temperature: float = 1.0,
        distribution: str = 'log_prob') -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions(rng, actor_def, actor_params, observations,
                           temperature, distribution)


class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


class NormalTanhDeterministicPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    dropout_rate: Optional[float] = None
    final_fc_init_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True
    init_mean: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 stddev: float = 0.2,
                 training: bool = False) -> tfd.Distribution:
        # x = MLP(self.hidden_dims,
        #               activate_final=True,
        #               dropout_rate=self.dropout_rate)(observations,
        #                                               training=training)
        x = observations
        for i, size in enumerate(self.hidden_dims):
            layer = nn.Dense(size, kernel_init=default_init(), name='dense{}'.format(i))
            x = layer(x)
            x = self.activations(x)
            x = IdentityLayer(name=f'{layer.name}_act')(x)

        means = nn.Dense(self.action_dim,
                         kernel_init=default_init(
                             self.final_fc_init_scale), name='final')(x)
        if self.init_mean is not None:
            means += self.init_mean
        means = nn.tanh(means)

        return TruncatedNormal(loc=means, scale=stddev)


class TruncatedNormal:
    """Truncated Normal distribution with values clipped to a range."""
    
    def __init__(self, loc: jnp.ndarray, scale: jnp.ndarray, 
                 low: float = -1.0, high: float = 1.0, eps: float = 1e-6):
        """Initialize the truncated normal distribution."""
        # Store the parameters specific to TruncatedNormal
        self.mu = jnp.clip(loc, low + eps, high - eps)
        self.sigma = jnp.zeros_like(loc) + scale # TODO schedule sigma as decribed in the paper
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x: jnp.ndarray) -> jnp.ndarray:
        """Clamp values while preserving gradients."""
        clamped_x = jnp.clip(x, self.low + self.eps, self.high - self.eps)
        return x - jax.lax.stop_gradient(x) + jax.lax.stop_gradient(clamped_x)

    def sample(self, seed: PRNGKey,
                    clip: Optional[float] = None,
                    sample_shape: Union[Tuple[int, ...], int] = ()) -> jnp.ndarray:
        """Sample from the truncated normal distribution with optional clipping of noise."""
        
        # Generate standard normal samples
        eps = jax.random.normal(seed, shape=self.mu.shape, dtype=self.mu.dtype)

        # Scale the samples
        eps = eps * self.sigma
        
        # Optionally clip the noise
        if clip is not None:
            eps = jnp.clip(eps, -clip, clip)

        # Add the mean
        x = self.mu + eps
        
        # Clamp the values
        return self._clamp(x)


@functools.partial(jax.jit, static_argnames=('actor_def', 'temperature'))
def _sample_deterministic_actions(
        rng: PRNGKey,
        stddev: float,
        actor_def: nn.Module,
        actor_params: Params,
        observations: np.ndarray,
        temperature: float) -> Tuple[PRNGKey, jnp.ndarray]:
    dist = actor_def.apply({'params': actor_params}, observations, stddev)
    if temperature == 0.:
        actions = dist.mu
    else:
        rng, key = jax.random.split(rng)
        actions = dist.sample(key, clip=None)
    return rng, actions


def sample_deterministic_actions(
        rng: PRNGKey,
        stddev: float,
        actor_def: nn.Module,
        actor_params: Params,
        observations: np.ndarray,
        temperature: float = 1.0) -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_deterministic_actions(rng, stddev, actor_def, actor_params, observations, temperature)
