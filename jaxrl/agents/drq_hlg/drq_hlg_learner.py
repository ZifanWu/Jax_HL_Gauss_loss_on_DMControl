"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.drq.augmentations import batched_random_crop
from jaxrl.agents.drq.networks import DrQDistributionalDoubleCritic, DrQDistributionalSingleCritic, DrQPolicy
from jaxrl.agents.sac import temperature
from jaxrl.agents.sac.critic import target_update
from jaxrl.datasets import Batch
from jaxrl.networks import policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey, ModelDecoupleOpt


# MIN_VALUE = 0
# MAX_VALUE = 100 # 1+0.99+0.99**2+...+0.99**1000=100

@functools.partial(jax.jit, 
                   static_argnames=('update_target', 'n_logits', 'sigma', 'batch_size', 'double_q', 'use_entropy'))
def _update_jit(
    n_logits: int, sigma: float, batch_size: int, double_q: bool, use_entropy: bool,
    min_value: float, max_value: float,
    rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
    temp: Model, batch: Batch, discount: float, tau: float,
    target_entropy: float, update_target: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)

    support = jnp.linspace(min_value, max_value, n_logits + 1, dtype=jnp.float32) # logits are centers! (ie, num of classes)
    centers = (support[:-1] + support[1:]) / 2
    support = support[None, :].repeat(batch_size, axis=0) # (B, n_logits+1)
    
    def transform_to_probs(target): # (B,)
        target = jnp.clip(target, min_value, max_value)
        # print(target.shape, support.shape) # (512) (B, n_logits+1)
        import time
        time.sleep(2)
        cdf_evals = jax.scipy.special.erf((support - target[:, None]) / (jnp.sqrt(2) * sigma)) # (B, n_logits+1)
        z = cdf_evals[:, -1] - cdf_evals[:, 0] # (B,)
        bin_probs = cdf_evals[:, 1:] - cdf_evals[:, :-1] # (B, n_logits)
        return bin_probs / z[:, None] # (B, n_logits)
    def transform_from_probs(probs):
        return (probs * centers).sum(-1) # (B, n_logits)
    
    if double_q:
        from jaxrl.agents.sac_hlg.critic import update as update_critic
        from jaxrl.agents.sac_hlg.actor import update as update_actor
    else:
        from jaxrl.agents.sac_hlg.critic_single import update as update_critic
        from jaxrl.agents.sac_hlg.actor_single import update as update_actor

    observations = batched_random_crop(key, batch.observations)
    rng, key = jax.random.split(rng)
    next_observations = batched_random_crop(key, batch.next_observations)

    batch = batch._replace(observations=observations,
                           next_observations=next_observations)

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(transform_to_probs, 
                                            transform_from_probs,
                                            use_entropy,
                                            key,
                                            actor,
                                            critic,
                                            target_critic,
                                            temp,
                                            batch,
                                            discount,
                                            soft_critic=True)
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    # Use critic conv layers in actor:
    new_actor_params = actor.params.copy(
        add_or_replace={'SharedEncoder': new_critic.params['SharedEncoder']})
    actor = actor.replace(params=new_actor_params)

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(transform_to_probs, 
                                        transform_from_probs,
                                        use_entropy,
                                        key, actor, new_critic, temp, batch)
    new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                              target_entropy)

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }


class DrQHLGaussianLearner(object):

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 n_logits: int = 51,
                 sigma: float=1.5,
                 min_value: float = 0.,
                 max_value: float = 100.,
                 batch_size: int=256,
                 double_q: bool = True,
                 use_entropy: bool = True,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 init_temperature: float = 0.1):

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_def = DrQPolicy(hidden_dims, action_dim, cnn_features,
                              cnn_strides, cnn_padding, latent_dim)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))

        if double_q:
            critic_def = DrQDistributionalDoubleCritic(hidden_dims, n_logits, cnn_features, 
                                                    cnn_strides, cnn_padding, latent_dim)
        else:
            critic_def = DrQDistributionalSingleCritic(hidden_dims, n_logits, cnn_features, 
                                                    cnn_strides, cnn_padding, latent_dim)
        # critic_def = DrQDoubleCritic(hidden_dims, cnn_features, cnn_strides,
        #                              cnn_padding, latent_dim)
        # critic = Model.create(critic_def,
        #                       inputs=[critic_key, observations, actions],
        #                       tx=optax.adam(learning_rate=critic_lr))
        critic = ModelDecoupleOpt.create(critic_def,
                                         inputs=[critic_key, observations, actions],
                                         tx=optax.adam(learning_rate=critic_lr),
                                         tx_enc=optax.adam(learning_rate=critic_lr))
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=optax.adam(learning_rate=temp_lr))

        self.n_logits = n_logits
        self.sigma = sigma
        self.batch_size = batch_size
        self.double_q = double_q
        self.use_entropy = use_entropy
        self.min_value = min_value
        self.max_value = max_value

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.rng = rng
        self.step = 0

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)

        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1
        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self.n_logits, self.sigma, self.batch_size, self.double_q, self.use_entropy,
            self.min_value, self.max_value,
            self.rng, self.actor, self.critic, self.target_critic, self.temp,
            batch, self.discount, self.tau, self.target_entropy,
            self.step % self.target_update_period == 0)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info
