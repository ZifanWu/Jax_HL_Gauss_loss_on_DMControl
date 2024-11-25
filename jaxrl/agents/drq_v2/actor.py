from typing import Tuple

import jax
import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey, tree_norm


# def update(key: PRNGKey, actor: Model, critic: Model, temp: Model,
#            batch: Batch) -> Tuple[Model, InfoDict]:

#     def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
#         dist = actor.apply_fn({'params': actor_params}, batch.observations)
#         actions = dist.sample(seed=key)
#         log_probs = dist.log_prob(actions)
#         q1, q2 = critic(batch.observations, actions)
#         q = jnp.minimum(q1, q2)
#         actor_loss = (log_probs * temp() - q).mean()
#         return actor_loss, {
#             'actor_loss': actor_loss,
#             'entropy': -log_probs.mean()
#         }

#     new_actor, info = actor.apply_gradient(actor_loss_fn)

#     return new_actor, info

def update(key: PRNGKey, stddev: float, encoder: Model, stddev_clip: float, actor: Model, critic: Model, 
           batch: Batch) -> Tuple[Model, InfoDict]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # encodings = encoder.apply({'params': encoder_params}, batch.observations)
        encodings = encoder(batch.observations)
        dist = actor.apply({'params': actor_params}, encodings, stddev)
        actions = dist.sample(seed=key, clip=stddev_clip)
        # actions = dist.sample(seed=key)
        q1, q2 = critic(encodings, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (- q).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'actor_pnorm': tree_norm(actor_params),
            'actor_action': jnp.mean(jnp.abs(actions))
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)
    info['actor_gnorm'] = info.pop('grad_norm')

    return new_actor, info
