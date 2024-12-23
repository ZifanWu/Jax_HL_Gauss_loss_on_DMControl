from typing import Tuple

import jax
from flax import linen as nn
import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey


def update(transform_to_probs,
           transform_from_probs,
           use_entropy,
           key: PRNGKey, actor: Model, critic: Model, temp: Model,
           batch: Batch) -> Tuple[Model, InfoDict]:

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist = actor.apply({'params': actor_params}, batch.observations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q_logits = critic(batch.observations, actions)
        q_probs = nn.softmax(q_logits)
        q = transform_from_probs(q_probs)
        if use_entropy:
            actor_loss = (log_probs * temp() - q).mean()
        else:
            actor_loss = (- q).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean()
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info
