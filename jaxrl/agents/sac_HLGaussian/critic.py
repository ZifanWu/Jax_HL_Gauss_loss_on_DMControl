from typing import Tuple

from flax import linen as nn
import jax.numpy as jnp
import optax

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey


def update(transform_to_probs, transform_from_probs,
           key: PRNGKey, actor: Model, critic: Model, target_critic: Model,
           temp: Model, batch: Batch, discount: float,
           backup_entropy: bool) -> Tuple[Model, InfoDict]:
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)
    next_q1_logits, next_q2_logits = target_critic(batch.next_observations, next_actions) # (B,n_logits)
    next_q1_probs, next_q2_probs = nn.softmax(next_q1_logits), nn.softmax(next_q2_logits)
    next_q1, next_q2 = transform_from_probs(next_q1_probs), transform_from_probs(next_q2_probs)

    next_q = jnp.minimum(next_q1, next_q2)
    # next_q_logits = 

    target_q = batch.rewards + discount * batch.masks * next_q
    target_probs = transform_to_probs(target_q)

    # loss = optax.softmax_cross_entropy(curr_probs, target_probs)

    if backup_entropy:
        target_q -= discount * batch.masks * temp() * next_log_probs

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1_logits, q2_logits = critic.apply_fn({'params': critic_params}, batch.observations,
                                 batch.actions)
        # q1_probs, q2_probs = nn.softmax(q1_logits), nn.softmax(q2_logits)
        # critic_loss = ((q1 - target_q)**2 + (q2 - target_q)**2).mean()
        critic_loss = optax.softmax_cross_entropy(q1_logits, target_probs) \
                    + optax.softmax_cross_entropy(q2_logits, target_probs)
        return critic_loss, {
            'critic_loss': critic_loss,
            # 'q1': q1_probs.mean(),
            # 'q2': q2_probs.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info
