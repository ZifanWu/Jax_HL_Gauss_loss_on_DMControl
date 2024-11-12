"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.sac import temperature
from jaxrl.agents.sac.critic import target_update

from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey, ModelDecoupleOptforLP
from jaxrl.agents.sequential_sac import weight_recyclers


# MIN_VALUE = 0
# MAX_VALUE = 100 # 1+0.99+0.99**2+...+0.99**1000=100

@functools.partial(jax.jit,
                   static_argnames=('soft_critic', 'update_target', 'n_logits', 'sigma', 'batch_size', 'double_q', 'use_entropy',
                                    'min_value', 'max_value'))
def _update_jit(n_logits: int, sigma: float, batch_size: int, double_q: bool, use_entropy: bool,
                min_value: float, max_value: float,
                rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
                temp: Model, batch: Batch, discount: float, tau: float,
                target_entropy: float, soft_critic: bool, update_target: bool
            ) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)

    support = jnp.linspace(min_value, max_value, n_logits + 1, dtype=jnp.float32) # logits are centers! (ie, num of classes)
    centers = (support[:-1] + support[1:]) / 2
    support = support[None, :].repeat(batch_size, axis=0) # (B, n_logits+1)
    
    def transform_to_probs(target): # (B,)
        target = jnp.clip(target, min_value, max_value)
        cdf_evals = jax.scipy.special.erf((support - target[:, None]) / (jnp.sqrt(2) * sigma)) # (B, n_logits+1)
        z = cdf_evals[:, -1] - cdf_evals[:, 0] # (B,)
        bin_probs = cdf_evals[:, 1:] - cdf_evals[:, :-1] # (B, n_logits)
        return bin_probs / z[:, None] # (B, n_logits)
    def transform_from_probs(probs):
        return (probs * centers).sum(-1) # (B, n_logits)
    
    if double_q:
        from jaxrl.agents.sequential_sac_hlg.critic import update as update_critic
        from jaxrl.agents.sequential_sac_hlg.actor import update as update_actor
    else:
        from jaxrl.agents.sequential_sac_hlg.critic_single import update as update_critic
        from jaxrl.agents.sequential_sac_hlg.actor_single import update as update_actor

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
                                            soft_critic=soft_critic)
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(transform_to_probs, 
                                         transform_from_probs, 
                                         use_entropy,
                                         key, actor, new_critic, temp, batch)
    new_temp, alpha_info = temperature.update(temp, actor_info['entropy'], target_entropy)

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }


class SequentialSACHLGLearner(object):

    def __init__(self,
                 seed: int,
                 track: bool,
                 replay_buffer,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 n_logits: int = 51,
                 sigma: float=1.5,
                 batch_size: int=256,
                 batch_size_statistics: int = 256,
                 dead_neurons_threshold: float = 0.1,
                 dormancy_logging_period: int = 2_000,
                 double_q: bool = True,
                 use_entropy: bool = True,
                 adam_eps: float = 1e-8,
                 min_value: float = 0.,
                 max_value: float = 100.,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 soft_critic: bool = True,
                 init_temperature: float = 1.0,
                 init_mean: Optional[np.ndarray] = None,
                 policy_final_fc_init_scale: float = 1.0):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.soft_critic = soft_critic

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
        actor_def = policies.NormalTanhPolicy(
            hidden_dims,
            action_dim,
            init_mean=init_mean,
            final_fc_init_scale=policy_final_fc_init_scale)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))

        if double_q:
            critic_def = critic_net.ActivationTrackLinearProjectorDistributionalDoubleCritic(hidden_dims, n_logits)
        else:
            critic_def = critic_net.ActivationTrackLinearProjectorDistributionalDoubleCritic(hidden_dims, n_logits, num_qs=1)
        # critic = Model.create(critic_def,
        #                       inputs=[critic_key, observations, actions],
        #                       tx=optax.adam(learning_rate=critic_lr, eps=adam_eps))
        critic = ModelDecoupleOptforLP.create(critic_def,
                                         inputs=[critic_key, observations, actions],
                                         tx=optax.adam(learning_rate=critic_lr),
                                         tx_critic=optax.adam(learning_rate=critic_lr))
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

        import flax
        param_dict = flax.traverse_util.flatten_dict(critic.params, sep='/')
        layer_list = list(param_dict.keys())[2:]
        layer_list = [l[:28] for l in layer_list]
        reset_layer_list = list(dict.fromkeys(layer_list))
        reset_layer_list = [l for l in reset_layer_list if 'final' not in l] #['reused_critic/critic0/dense0', 'reused_critic/critic0/dense1', 'reused_critic/critic1/dense0', 'reused_critic/critic1/dense1']
        self.weight_recycler = weight_recyclers.BaseRecycler(reset_layer_list, track, dead_neurons_threshold, dormancy_logging_period=dormancy_logging_period)

        self.replay_buffer = replay_buffer
        self.batch_size_statistics = batch_size_statistics

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.rng = rng

        self.step = 1

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)
    
    def get_intermediates(self, network, online_params):
        batch = self.replay_buffer.sample(self.batch_size_statistics)
        _, state = network.apply(
            {'params': online_params},
            batch.observations,
            batch.actions,
            capture_intermediates=lambda l, _: l.name is not None and 'act' in l.name,
            mutable=['intermediates'],
        )
        return state['intermediates']

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1

        critic_online_params, actor_online_params = self.critic.params, self.actor.params
        is_intermediated = self.weight_recycler.is_intermediated_required(
            self.step
        )
        critic_intermediates = (
            self.get_intermediates(self.critic, critic_online_params) if is_intermediated else None
        )
        self.weight_recycler.maybe_log_deadneurons(
            self.step, critic_intermediates
        )
        # actor_intermediates = (
        #     self.get_intermediates(self.actor, actor_online_params) if is_intermediated else None
        # )
        # self.weight_recycler.maybe_log_deadneurons(
        #     self.step, actor_intermediates
        # )

        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self.n_logits, self.sigma, self.batch_size, self.double_q, self.use_entropy,
            self.min_value, self.max_value,
            self.rng, self.actor, self.critic, self.target_critic, self.temp,
            batch, self.discount, self.tau, self.target_entropy,
            self.soft_critic, self.step % self.target_update_period == 0)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info
