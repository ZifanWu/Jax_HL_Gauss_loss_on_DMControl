"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple
import re

import flax.traverse_util
from optax._src import base
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax

from jaxrl.agents.drq.augmentations import batched_random_crop
from jaxrl.agents.drq.networks import DrQDoubleCritic, DrQPolicy, ActivationTrackDrQDoubleCritic
from jaxrl.agents.sac import temperature
from jaxrl.agents.sac.actor import update as update_actor
from jaxrl.agents.drq_weight_pruning.critic import target_update
from jaxrl.agents.drq_weight_pruning.critic import update as update_critic
from jaxrl.datasets import Batch
from jaxrl.networks import policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey, ModelDecoupleOpt
from jaxrl.agents.drq import weight_recyclers
import jaxpruner
from jaxpruner.algorithms.pruners import MagnitudePruning
from jaxrl.agents.drq_weight_pruning.sparse_util import create_updater_from_config


@functools.partial(jax.jit, static_argnames=('update_target'))
def _update_jit_fn(
    rng: PRNGKey, actor: Model, critic: ModelDecoupleOpt, target_critic: Model,
    temp: Model, batch: Batch, discount: float, tau: float,
    target_entropy: float, update_target: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    observations = batched_random_crop(key, batch.observations)
    rng, key = jax.random.split(rng)
    next_observations = batched_random_crop(key, batch.next_observations)

    batch = batch._replace(observations=observations,
                           next_observations=next_observations)

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

    # Use critic conv layers in actor:
    new_actor_params = actor.params.copy(
        add_or_replace={'SharedEncoder': new_critic.params['SharedEncoder']})
    actor = actor.replace(params=new_actor_params)

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)
    new_temp, alpha_info = temperature.update(temp, actor_info['entropy'],
                                              target_entropy)

    return rng, new_actor, new_critic, new_target_critic, new_temp, {
        **critic_info,
        **actor_info,
        **alpha_info
    }


class DrQWPLearner(object):

    def __init__(self,
                 seed: int,
                 track: bool,
                 replay_buffer,
                 sparse_reward: bool,
                 sparse_steps: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 delta: float = 0.01,
                 prune_start_step: int = 4e5,
                 prune_end_step: int = 16e5,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 actor_hidden_dims: Sequence[int] = (256, 256),
                 critic_hidden_dims: Sequence[int] = (256, 256),
                 batch_size: int = 512,
                 batch_size_statistics: int = 256,
                 dead_neurons_thresholds: Sequence[float] = [0., 0.025, 0.1],
                 dormancy_logging_period: int = 2_000,
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 init_temperature: float = 0.1):

        action_dim = actions.shape[-1] # q-r: 12 h-h: 4
        # print(action_dim, observations.shape) # (1, 84, 84, 9)

        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key, pruner_rng = jax.random.split(rng, 5)

        actor_def = DrQPolicy(actor_hidden_dims, action_dim, cnn_features,
                              cnn_strides, cnn_padding, latent_dim)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))

        # critic_def = DrQDoubleCritic(hidden_dims, cnn_features, cnn_strides,
        #                                             cnn_padding, latent_dim)
        critic_def = ActivationTrackDrQDoubleCritic(critic_hidden_dims, cnn_features, cnn_strides,
                                                    cnn_padding, latent_dim)
        # critic = Model.create(critic_def,
        #                       inputs=[critic_key, observations, actions],
        #                       tx=optax.adam(learning_rate=critic_lr))
        sparsity_distribution = functools.partial(
            jaxpruner.sparsity_distributions.uniform, sparsity=0.95)
        self.pruner = MagnitudePruning(sparsity_distribution_fn=sparsity_distribution,
                                                 scheduler=jaxpruner.sparsity_schedules.PolynomialSchedule(
                                                    update_freq=1000, update_start_step=prune_start_step, update_end_step=prune_end_step)
                                                )
        # self.pruner = create_updater_from_config(rng_seed=pruner_rng)
        self.post_gradient_update = jax.jit(self.pruner.post_gradient_update)
        critic_head_optimizer = self.pruner.wrap_optax(optax.adam(learning_rate=critic_lr))
        # encoder_optimizer = self.pruner.wrap_optax(optax.adam(learning_rate=critic_lr))

        critic = ModelDecoupleOpt.create(critic_def,
                                         inputs=[critic_key, observations, actions],
                                         tx=critic_head_optimizer,
                                         tx_enc=optax.adam(learning_rate=critic_lr))
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
        self.step = 0

        import flax
        def get_layer_list(model: Model) -> list[str]:
            param_dict = flax.traverse_util.flatten_dict(model.params, sep='/')
            layer_list = list(param_dict.keys())
            # print(1111, layer_list)
            layer_list = [l[:l.rfind('/')] for l in layer_list]
            # print(2222, layer_list)
            layer_list = list(dict.fromkeys(layer_list))
            # print(3333, layer_list)
            # layer_list = [l for l in layer_list if 'final' not in l and l != '']
            layer_list = [l for l in layer_list if ('dense' in l or 'final' in l) and 'layernorm' not in l]
            print(4444, layer_list)
            return layer_list

        critic_layer_list = get_layer_list(critic)
        actor_layer_list = get_layer_list(actor)
        critic1_layer_list = [l for l in critic_layer_list if 'critic0' in l]
        critic2_layer_list = [l for l in critic_layer_list if 'critic1' in l]
        self.critic1_weight_recycler = weight_recyclers.BaseRecycler(critic1_layer_list, 
                                                                    track=track, 
                                                                    dead_neurons_thresholds=dead_neurons_thresholds, 
                                                                    dormancy_logging_period=dormancy_logging_period,
                                                                    delta=delta)
        self.actor_weight_recycler = weight_recyclers.BaseRecycler(actor_layer_list, 
                                                                    track, 
                                                                    dead_neurons_thresholds=dead_neurons_thresholds, 
                                                                    dormancy_logging_period=dormancy_logging_period, 
                                                                    delta=delta
                                                                    )

        self.replay_buffer = replay_buffer
        self.batch_size_statistics = batch_size_statistics
        self.sparse_reward = sparse_reward
        self.sparse_steps = sparse_steps

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)

        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)
    
    def get_critic_intermediates(self, network, online_params):
        batch = self.replay_buffer.sample(self.batch_size_statistics)
        def filter_rep(l, _):
            return (l.name is not None and 
                    ('_act' in l.name or '_preact' in l.name))
        _, state = network.apply(
            {'params': online_params},
            batch.observations,
            batch.actions,
            capture_intermediates=filter_rep,#lambda l, _: l.name is not None and 'act' in l.name,
            mutable=['intermediates'],
        )
        # return state['intermediates']
        intermediates = state['intermediates']
        intermediates = flax.traverse_util.flatten_dict(intermediates, sep='/')
        # print(3424, intermediates.keys())#['SharedEncoder', 'dense-1_layernorm_tanh_preact', 'dense-1_layernorm_tanh_act', 'CriticHead']
        # print(432, intermediates['CriticHead'].keys())['critic0', 'critic1']
        # print(3242, intermediates['CriticHead']['critic0'].keys())
        # import time
        # time.sleep(222)
        activations = {k: v for k, v in intermediates.items() if '_act' in k and 'conv' not in k}
        preactivations = {k: v for k, v in intermediates.items() if '_preact' in k and 'conv' not in k}

        return activations, preactivations
    
    def get_actor_intermediates(self, network, online_params):
        batch = self.replay_buffer.sample(self.batch_size_statistics)
        def filter_rep(l, _):
            return (l.name is not None and 
                    ('_act' in l.name or '_preact' in l.name))
        _, state = network.apply(
            {'params': online_params},
            batch.observations,
            capture_intermediates=filter_rep,#lambda l, _: l.name is not None and 'act' in l.name,
            mutable=['intermediates'],
        )
        # return state['intermediates']
        intermediates = state['intermediates']
        intermediates = flax.traverse_util.flatten_dict(intermediates, sep='/')
        activations = {k: v for k, v in intermediates.items() if '_act' in k and 'conv' not in k}
        preactivations = {k: v for k, v in intermediates.items() if '_preact' in k and 'conv' not in k}

        return activations, preactivations

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1
        if self.sparse_reward and self.step <= self.sparse_steps:
            batch = batch._replace(rewards=np.zeros_like(batch.rewards))
        
        # _update_jit_fn = jax.jit(
        #     functools.partial(
        #         _update, pruner=self.pruner, update_target=self.step%self.target_update_period == 0
        #         )
        #     )

        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit_fn(
            self.rng, self.actor, self.critic, self.target_critic, self.temp,
            batch, self.discount, self.tau, self.target_entropy, self.step%self.target_update_period == 0)
        
        new_critic_params = {}
        new_critic_params['SharedEncoder'] = new_critic.params['SharedEncoder']
        new_critichead_params = flax.core.FrozenDict({k: v for k, v in new_critic.params.items() if 'Encoder' not in k})

        new_critichead_params = self.post_gradient_update(new_critichead_params, new_critic.opt_state_head)
        
        for k, v in new_critichead_params.items():
            new_critic_params[k] = v

        new_critic = new_critic.replace(params=flax.core.FrozenDict(new_critic_params))

        is_intermediated = self.critic1_weight_recycler.is_intermediated_required(self.step)
        critic_intermediates, critic_preacts = (
            self.get_critic_intermediates(new_critic, new_critic.params) if is_intermediated else (None, None)
        )

        if is_intermediated:
            critic1_intermediates = {k: v for k, v in critic_intermediates.items() if 'critic0' in k}
            critic2_intermediates = {k: v for k, v in critic_intermediates.items() if 'critic1' in k}
            critic1_preacts = {k: v for k, v in critic_preacts.items() if 'critic0' in k}
            critic2_preacts = {k: v for k, v in critic_preacts.items() if 'critic1' in k}
        else:
            critic1_intermediates, critic2_intermediates, critic1_preacts, critic2_preacts = [None] * 4
        # print(critic1_intermediates==None, critic2_intermediates==None, critic1_preacts==None, critic2_preacts==None)
        self.critic1_weight_recycler.maybe_log_deadneurons(
            self.step, critic1_intermediates, critic1_preacts, new_critic.params
        ) # step-1: we log the first step's deadneurons

        actor_intermediates, actor_preacts = (
            self.get_actor_intermediates(new_actor, new_actor.params) if is_intermediated else (None, None)
        )
        self.actor_weight_recycler.maybe_log_deadneurons(
            self.step, actor_intermediates, actor_preacts, new_actor.params
        )

        self.rng = new_rng

        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info
