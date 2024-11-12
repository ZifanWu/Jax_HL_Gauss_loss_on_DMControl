"""Implementations of algorithms for continuous control."""

# from typing import Callable, Sequence, Tuple
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jax.numpy as jnp
from flax import linen as nn

from jaxrl.networks.common import MLP


class ValueCritic(nn.Module):
    hidden_dims: Sequence[int]

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, 1))(observations)
        return jnp.squeeze(critic, -1)


class Critic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, 1),
                     activations=self.activations)(inputs)
        return jnp.squeeze(critic, -1)


class DoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2

    @nn.compact
    def __call__(self, states, actions):

        VmapCritic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.num_qs)
        qs = VmapCritic(self.hidden_dims,
                        activations=self.activations)(states, actions)
        return qs


class LinearProjector(nn.Module):
    hidden_dim: int
    # activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions], -1)
        x = MLP((self.hidden_dim,), name='linear_projector')(x)
        return x


class ActivationTrackCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    n_logits: int = 1
    activate_final: int = False
    dropout_rate: Optional[float] = None
    layer_names = []

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        is_init = not self.has_variable('params', 'dense0')
        if is_init:
            for _ in range(len(self.layer_names)):
                self.layer_names.pop()
        for i, size in enumerate(self.hidden_dims):
            layer = nn.Dense(size, kernel_init=default_init(), name='dense{}'.format(i))
            self.layer_names.append(layer.name)
            x = layer(x)
            x = self.activations(x)
            x = IdentityLayer(name=f'{layer.name}_act')(x)
            if self.dropout_rate is not None:
                x = nn.Dropout(rate=self.dropout_rate)(
                    x, deterministic=not training)
        layer = nn.Dense(self.n_logits, kernel_init=default_init(), name='final')
        x = layer(x)
        if self.activate_final:
            x = self.activations(x)
            if self.dropout_rate is not None:
                x = nn.Dropout(rate=self.dropout_rate)(
                    x, deterministic=not training)
            x = IdentityLayer(name=f'{layer.name}_act')(x)
        return x


class ActivationTrackDoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    num_qs: int = 2

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        critics = []
        for q in range(self.num_qs):
            x = inputs
            critic = ActivationTrackCritic(self.hidden_dims, self.activations,
                              name='critic{}'.format(q))(x)
            critics.append(critic)
        return jnp.stack(critics) # (2, 1, 1)


class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


def default_init(scale: Optional[float] = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class ActivationTrackLinearProjectorDoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    num_qs: int = 2
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, states, actions):
        x = jnp.concatenate([states, actions], -1)
        layer = nn.Dense(self.hidden_dims[0], kernel_init=default_init(), name='linear_projector')
        inputs = layer(x)
        critic = ActivationTrackDoubleCritic(self.hidden_dims, self.activations,
                                           self.num_qs, name='reused_critic')
        q_values = critic(inputs)
        return q_values


class DistributionalCritic(nn.Module):

    hidden_dims: Sequence[int]
    n_logits: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], -1)
        critic = MLP((*self.hidden_dims, self.n_logits),
                     activations=self.activations)(inputs)
        return critic


class DoubleDistributionalCritic(nn.Module):
    hidden_dims: Sequence[int]
    n_logits: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2

    @nn.compact
    def __call__(self, states, actions):

        VmapCritic = nn.vmap(DistributionalCritic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.num_qs)
        qs = VmapCritic(self.hidden_dims,
                        self.n_logits,
                        activations=self.activations)(states, actions)
        return qs


class LPDistributionalCritic(nn.Module):
    hidden_dims: Sequence[int]
    n_logits: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        critic = MLP((*self.hidden_dims, self.n_logits),
                     activations=self.activations, name='ReusedCritic')(x)
        return critic


class LPDoubleDistributionalCritic(nn.Module):
    hidden_dims: Sequence[int]
    n_logits: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    num_qs: int = 2

    @nn.compact
    def __call__(self, states, actions):
        x = LinearProjector(self.hidden_dims[0],
                            name='LinearProjector')(states, actions)
        VmapCritic = nn.vmap(LPDistributionalCritic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=0,
                             axis_size=self.num_qs)
        qs = VmapCritic(self.hidden_dims,
                        self.n_logits,
                        activations=self.activations, name='ReusedCritic')(x)
        return qs


class ActivationTrackDistributionalDoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    n_logits: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    num_qs: int = 2

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        critics = []
        for q in range(self.num_qs):
            x = inputs
            critic = ActivationTrackCritic(self.hidden_dims, self.activations, self.n_logits,
                              name='critic{}'.format(q))(x)
            critics.append(critic)
        return jnp.stack(critics)


class ActivationTrackLinearProjectorDistributionalDoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    n_logits: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    num_qs: int = 2
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, states, actions):
        x = jnp.concatenate([states, actions], -1)
        layer = nn.Dense(self.hidden_dims[0], kernel_init=default_init(), name='linear_projector')
        inputs = layer(x)
        critic = ActivationTrackDistributionalDoubleCritic(self.hidden_dims, self.n_logits, self.activations,
                                           self.num_qs, name='reused_critic')
        q_values = critic(inputs)
        return q_values