from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

from jaxrl.networks.common import default_init
from jaxrl.networks.critic_net import DoubleCritic, ActivationTrackDoubleDistributionalCritic, DistributionalCritic, ActivationTrackDoubleCritic
from jaxrl.networks.policies import NormalTanhPolicy, NormalTanhDeterministicPolicy


class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


class Encoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = 'VALID'
    use_batch_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        x = observations.astype(jnp.float32) / 255.0
        layer_count = 0
        for features, stride in zip(self.features, self.strides):
            layer = nn.Conv(features,
                        kernel_size=(3, 3),
                        strides=(stride, stride),
                        kernel_init=default_init(),
                        padding=self.padding,
                        name='conv{}'.format(layer_count))
            x = layer(x)
            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=not train)(observations)
            x = nn.relu(x)
            # x = IdentityLayer(name=f'{layer.name}_act')(x)
            layer_count += 1

        if len(x.shape) == 4:
            x = x.reshape([x.shape[0], -1])
        else:
            x = x.reshape([-1])
        return x


class DrQDoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = 'VALID'
    latent_dim: int = 50

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = Encoder(self.cnn_features,
                    self.cnn_strides,
                    self.cnn_padding,
                    name='SharedEncoder')(observations)

        layer = nn.Dense(self.latent_dim)
        x = layer(x)
        # x = IdentityLayer(name=f'{layer.name}_preact')(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        # x = IdentityLayer(name=f'{layer.name}_act')(x)

        return DoubleCritic(self.hidden_dims)(x, actions)


class ActivationTrackDrQDoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = 'VALID'
    latent_dim: int = 50
    use_LN: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = Encoder(self.cnn_features,
                    self.cnn_strides,
                    self.cnn_padding,
                    name='SharedEncoder')(observations)
        layer = nn.Dense(self.latent_dim, name='dense-1_layernorm_tanh')
        x = layer(x)
        # x = IdentityLayer(name=f'{layer.name}_preact')(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        # x = IdentityLayer(name=f'{layer.name}_act')(x)
        x = jnp.concatenate([x, actions], -1)

        return ActivationTrackDoubleCritic(self.hidden_dims, name='CriticHead', use_LN=self.use_LN)(x)


class ActivationTrackDrQDistributionalDoubleCritic(nn.Module):
    hidden_dims: Sequence[int]
    n_logits: int
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = 'VALID'
    latent_dim: int = 50
    num_qs: int = 2
    use_layer_norm: bool = False
    use_batch_norm: bool = False

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray, train: bool=False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = Encoder(self.cnn_features,
                    self.cnn_strides,
                    self.cnn_padding,
                    name='SharedEncoder',
                    use_batch_norm=self.use_batch_norm)(observations, train)

        layer = nn.Dense(self.latent_dim, name='dense-1_layernorm_tanh')
        x = layer(x)
        x = IdentityLayer(name=f'{layer.name}_preact')(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        x = IdentityLayer(name=f'{layer.name}_act')(x)
        x = jnp.concatenate([x, actions], -1)

        return ActivationTrackDoubleDistributionalCritic(self.hidden_dims, self.n_logits, 
                                                         num_qs=self.num_qs, name='CriticHead', 
                                                         use_layer_norm=self.use_layer_norm)(x)


class DrQDistributionalSingleCritic(nn.Module):
    hidden_dims: Sequence[int]
    n_logits: int
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = 'VALID'
    latent_dim: int = 50

    @nn.compact
    def __call__(self, observations: jnp.ndarray,
                 actions: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = Encoder(self.cnn_features,
                    self.cnn_strides,
                    self.cnn_padding,
                    name='SharedEncoder')(observations)

        x = nn.Dense(self.latent_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)

        return DistributionalCritic(self.hidden_dims, self.n_logits, name='CriticHead')(x, actions)


class DrQPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = 'VALID'
    latent_dim: int = 50
    use_batch_norm: bool = False

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 train: bool = False) -> tfd.Distribution:
        x = Encoder(self.cnn_features,
                    self.cnn_strides,
                    self.cnn_padding,
                    name='SharedEncoder',
                    use_batch_norm=self.use_batch_norm)(observations, train)

        # We do not update conv layers with policy gradients.
        x = jax.lax.stop_gradient(x)

        layer = nn.Dense(self.latent_dim, name='dense-1_layernorm_tanh')
        x = layer(x)
        x = IdentityLayer(name=f'{layer.name}_preact')(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        x = IdentityLayer(name=f'{layer.name}_act')(x)

        return NormalTanhPolicy(self.hidden_dims, self.action_dim)(x,
                                                                   temperature)
