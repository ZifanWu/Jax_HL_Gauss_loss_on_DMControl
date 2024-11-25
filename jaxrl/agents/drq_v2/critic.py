from typing import Tuple

import jax
import flax
import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey, tree_norm


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_multimap(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)
    # new_target_params = flax.core.unfreeze(new_target_params)
    # new_target_params['SharedEncoder'] = target_critic.params['SharedEncoder']
    # new_target_params = flax.core.freeze(new_target_params)

    return target_critic.replace(params=new_target_params)


# def update(key: PRNGKey, encoder: Model, actor: Model, critic: Model, target_critic: Model,
#             batch: Batch, discount: float, stddev_clip: float,
#            soft_critic: bool) -> Tuple[Model, InfoDict]:
#     next_encodings = encoder(batch.next_observations)
#     dist = actor(next_encodings)
#     next_actions = dist.sample(seed=key, clip=stddev_clip)
#     # next_actions = dist.sample(seed=key)
#     # next_log_probs = dist.log_prob(next_actions)
#     next_qs = target_critic(batch.next_observations, next_actions) # (2, B)
#     next_q1, next_q2 = next_qs[0], next_qs[1]
#     next_q = jnp.minimum(next_q1, next_q2)

#     target_q = batch.rewards + discount * batch.masks * next_q

#     # if soft_critic:
#     #     target_q -= discount * batch.masks * temp() * next_log_probs

#     def critic_loss_fn(encoder_params: Params, critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
#         critic_fn = lambda actions: critic.apply({'params': critic_params}, 
#                                                  batch.observations, actions)
#         def _critic_fn(actions):
#             qs = critic_fn(actions)
#             q1, q2 = qs[0], qs[1]
#             return 0.5*(q1 + q2).mean(), (q1, q2)

#         (_, (q1, q2)), action_grad = jax.value_and_grad(_critic_fn, 
#                                                         has_aux=True)(
#                                                             batch.actions)
#         critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
#         return critic_loss, {
#             'critic_loss': critic_loss,
#             'q1': q1.mean(),
#             'q2': q2.mean(),
#             'r': batch.rewards.mean(),
#             'critic_pnorm': tree_norm(critic_params),
#             'critic_agnorm': jnp.sqrt((action_grad ** 2).sum(-1)).mean(0)
#         }

#     new_critic, info = critic.apply_gradient(critic_loss_fn)
#     info['critic_gnorm'] = info.pop('grad_norm')

#     return new_critic, info

def update(key: PRNGKey, stddev: float, encoder: Model, actor: Model, critic: Model, target_critic: Model,
            batch: Batch, discount: float, stddev_clip: float,
           soft_critic: bool) -> Tuple[Model, InfoDict]:
    next_encodings = encoder(batch.next_observations)
    dist = actor(next_encodings, stddev)
    next_actions = dist.sample(seed=key, clip=stddev_clip)
    next_qs = target_critic(next_encodings, next_actions) # (2, B)
    next_q1, next_q2 = next_qs[0], next_qs[1]
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = batch.rewards + discount * batch.masks * next_q

    current_encodings = encoder(batch.observations)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        # encodings = encoder.apply({'params': encoder_params}, batch.observations)
        # encodings = encoder(batch.observations)
        qs = critic.apply({'params': critic_params}, current_encodings, batch.actions)
        q1, q2 = qs[0], qs[1]
        
        critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean(),
            'r': batch.rewards.mean(),
        }
    def encoder_loss_fn(encoder_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        encodings = encoder.apply({'params': encoder_params}, batch.observations)
        # encodings = encoder(batch.observations)
        qs = critic(encodings, batch.actions)
        q1, q2 = qs[0], qs[1]
        
        encoder_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        return encoder_loss, {
            'encoder_loss': encoder_loss,
        }
    new_encoder, encoder_info = encoder.apply_gradient(encoder_loss_fn)
    new_critic, critic_info = critic.apply_gradient(critic_loss_fn)

    return new_encoder, new_critic, {**encoder_info, **critic_info}