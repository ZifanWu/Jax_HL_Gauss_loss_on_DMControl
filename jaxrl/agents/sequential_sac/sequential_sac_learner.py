"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.sac import temperature
from jaxrl.agents.sequential_sac.actor import update as update_actor
from jaxrl.agents.sequential_sac.critic import target_update
from jaxrl.agents.sequential_sac.critic import update as update_critic
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey, ModelDecoupleOptforLP
from jaxrl.agents.sequential_sac import weight_recyclers


@functools.partial(jax.jit,
                   static_argnames=('soft_critic', 'update_target')) # TODO turn it on after debugging!
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
    temp: Model, batch: Batch, discount: float, tau: float,
    target_entropy: float, soft_critic: bool, update_target: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    new_critic, critic_info = update_critic(key,
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

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
    new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                              target_entropy)

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }


class SequentialSACLearner(object):

    def __init__(self,
                 seed: int,
                 track: bool,
                 replay_buffer,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 global_step: int = 1,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 batch_size: int = 256,
                 batch_size_statistics: int = 256,
                 dead_neurons_threshold: float = 0.1,
                 dormancy_logging_period: int = 2_000,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 backup_entropy: bool = True,
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

        self.backup_entropy = backup_entropy

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

        critic_def = critic_net.ActivationTrackLinearProjectorDoubleCritic(hidden_dims)
        # critic = Model.create(critic_def,
        #                       inputs=[critic_key, observations, actions],
        #                       tx=optax.adam(learning_rate=critic_lr))
        critic = ModelDecoupleOptforLP.create(critic_def,
                                         inputs=[critic_key, observations, actions],
                                         tx=optax.adam(learning_rate=critic_lr),
                                         tx_critic=optax.adam(learning_rate=critic_lr))
        # critic.params: {'linear_projector': jnp.array, 'reused_critic': {'critic0': {'dense0': jnp.array, 'dense1': ...}, 'critic1': {...}}}
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=optax.adam(learning_rate=temp_lr))

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.rng = rng

        import flax
        param_dict = flax.traverse_util.flatten_dict(critic.params, sep='/')
        layer_list = list(param_dict.keys())[2:]
        layer_list = [l[:28] for l in layer_list]
        reset_layer_list = list(dict.fromkeys(layer_list))
        reset_layer_list = [l for l in reset_layer_list if 'final' not in l] #['reused_critic/critic0/dense0', 'reused_critic/critic0/dense1', 'reused_critic/critic1/dense0', 'reused_critic/critic1/dense1']
        self.weight_recycler = weight_recyclers.BaseRecycler(reset_layer_list, track, dead_neurons_threshold, dormancy_logging_period=dormancy_logging_period)

        self.replay_buffer = replay_buffer
        self.batch_size_statistics = batch_size_statistics

        self.step = global_step

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
            self.rng, self.actor, self.critic, self.target_critic, self.temp,
            batch, self.discount, self.tau, self.target_entropy,
            self.backup_entropy, self.step % self.target_update_period == 0)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info
