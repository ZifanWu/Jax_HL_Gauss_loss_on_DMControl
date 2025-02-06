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
from jaxrl.agents.sac.critic import target_update
from jaxrl.agents.sac.critic import update as update_critic
from jaxrl.datasets import Batch
from jaxrl.networks import policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey, ModelDecoupleOpt
from jaxrl.agents.drq import weight_recyclers


@functools.partial(jax.jit, static_argnames=('update_target'))
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
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


class DrQLearner(object):

    def __init__(self,
                 seed: int,
                 track: bool,
                 replay_buffer,
                 redo_critic: bool,
                 redo_actor: bool,
                 neutralize_dormant_neurons: bool,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 reset_interval: int,
                 reset_start_step: int,
                 use_WD_in_critic: bool = False,
                 use_LNWD_in_critic: bool = False,
                 WD_rate: float = 0.001,
                 reset_mass_opt_state: bool = False,
                 ntrlize_thres: float = 2.,
                 NO_K_mass_thres: bool = True,
                 weight_scaling: bool = False,
                 incoming_scale: float = 10.0,
                 K: int = 5,
                 mass_thres: float = 10.,
                 dead_thres: float = 0.1,
                 weight_revive_eps: float = 0.01,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
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
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor_def = DrQPolicy(hidden_dims, action_dim, cnn_features,
                              cnn_strides, cnn_padding, latent_dim)
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))

        # critic_def = DrQDoubleCritic(hidden_dims, cnn_features, cnn_strides,
        #                                             cnn_padding, latent_dim)
        critic_def = ActivationTrackDrQDoubleCritic(hidden_dims, cnn_features, cnn_strides,
                                                    cnn_padding, latent_dim,
                                                    use_LN=use_LNWD_in_critic)
        # critic = Model.create(critic_def,
        #                       inputs=[critic_key, observations, actions],
        #                       tx=optax.adam(learning_rate=critic_lr))
        if use_LNWD_in_critic or use_WD_in_critic:
            enc_optimizer = optax.adamw(learning_rate=critic_lr, weight_decay=WD_rate)
            head_optimizer = optax.adamw(learning_rate=critic_lr, weight_decay=WD_rate)
        else:
            enc_optimizer = optax.adam(learning_rate=critic_lr)
            head_optimizer = optax.adam(learning_rate=critic_lr)
        critic = ModelDecoupleOpt.create(critic_def,
                                         inputs=[critic_key, observations, actions],
                                         tx=enc_optimizer,
                                         tx_enc=head_optimizer)
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
            print('layer name list: ', layer_list)
            return layer_list

        critic_layer_list = get_layer_list(critic)
        actor_layer_list = get_layer_list(actor)
        critic1_layer_list = [l for l in critic_layer_list if 'critic0' in l]
        critic2_layer_list = [l for l in critic_layer_list if 'critic1' in l]
        if redo_critic:
            self.critic1_weight_recycler = weight_recyclers.NeuronRecycler(critic1_layer_list, 
                                                                        track=track,
                                                                        reset_period=reset_interval,
                                                                        dead_neurons_thresholds=dead_neurons_thresholds, 
                                                                        dormancy_logging_period=dormancy_logging_period,
                                                                        neutralize_dormant_neurons=neutralize_dormant_neurons,
                                                                        dead_thres=dead_thres, mass_thres=mass_thres,
                                                                        weight_revive_eps=weight_revive_eps,
                                                                        K=K,
                                                                        reset_start_step=reset_start_step,
                                                                        NO_K_mass_thres=NO_K_mass_thres,
                                                                        ntrlize_thres=ntrlize_thres,
                                                                        reset_mass_opt_state=reset_mass_opt_state,
                                                                        )
            self.critic2_weight_recycler = weight_recyclers.NeuronRecycler(critic2_layer_list, 
                                                                        track=track, 
                                                                        reset_period=reset_interval,
                                                                        dead_neurons_thresholds=dead_neurons_thresholds, 
                                                                        dormancy_logging_period=dormancy_logging_period,
                                                                        neutralize_dormant_neurons=neutralize_dormant_neurons,
                                                                        dead_thres=dead_thres, mass_thres=mass_thres,
                                                                        weight_revive_eps=weight_revive_eps,
                                                                        K=K,
                                                                        reset_start_step=reset_start_step,
                                                                        NO_K_mass_thres=NO_K_mass_thres,
                                                                        ntrlize_thres=ntrlize_thres,
                                                                        reset_mass_opt_state=reset_mass_opt_state,
                                                                        )
        else:
            self.critic1_weight_recycler = weight_recyclers.BaseRecycler(critic1_layer_list, 
                                                                        track=track, 
                                                                        dead_neurons_thresholds=dead_neurons_thresholds, 
                                                                        dormancy_logging_period=dormancy_logging_period)
        if redo_actor:
            self.actor_weight_recycler = weight_recyclers.NeuronRecycler(actor_layer_list, 
                                                                        track=track, 
                                                                        reset_period=reset_interval,
                                                                        dead_neurons_thresholds=dead_neurons_thresholds, 
                                                                        dormancy_logging_period=dormancy_logging_period,
                                                                        neutralize_dormant_neurons=neutralize_dormant_neurons,
                                                                        dead_thres=dead_thres, mass_thres=mass_thres,
                                                                        weight_revive_eps=weight_revive_eps,
                                                                        K=K,
                                                                        reset_start_step=reset_start_step,
                                                                        NO_K_mass_thres=NO_K_mass_thres,
                                                                        ntrlize_thres=ntrlize_thres,
                                                                        reset_mass_opt_state=reset_mass_opt_state,
                                                                        )
        else:
            self.actor_weight_recycler = weight_recyclers.BaseRecycler(actor_layer_list, 
                                                                        track, 
                                                                        dead_neurons_thresholds=dead_neurons_thresholds, 
                                                                        dormancy_logging_period=dormancy_logging_period, 
                                                                        )

        self.replay_buffer = replay_buffer
        self.batch_size_statistics = batch_size_statistics
        self.redo_critic = redo_critic
        self.redo_actor = redo_actor

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

        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self.rng, self.actor, self.critic, self.target_critic, self.temp,
            batch, self.discount, self.tau, self.target_entropy,
            self.step % self.target_update_period == 0)
        
        is_intermediated = self.critic1_weight_recycler.is_intermediated_required(self.step)
        critic_intermediates, critic_preacts = (
            self.get_critic_intermediates(new_critic, new_critic.params) if is_intermediated else (None, None)
        )
        # print(is_intermediated)
        # print(critic_intermediates == None)
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
        # self.critic2_weight_recycler.maybe_log_deadneurons(
        #     self.step-1, critic2_intermediates, critic2_preacts, new_critic.params['CriticHead']['critic0']
        # ) # step-1: we log the first step's deadneurons
        actor_intermediates, actor_preacts = (
            self.get_actor_intermediates(new_actor, new_actor.params) if is_intermediated else (None, None)
        )
        self.actor_weight_recycler.maybe_log_deadneurons(
            self.step, actor_intermediates, actor_preacts, new_actor.params
        )
        # print(new_critic.params.keys())frozen_dict_keys(['CriticHead', 'LayerNorm_0', 'SharedEncoder', 'dense-1_layernorm_tanh'])
        # for k in new_critic.params.keys():
        #     print(new_critic.params[k].keys())
        # import time
        # time.sleep(222)
        self.rng = new_rng
        if self.redo_critic:
            self.rng, key = jax.random.split(self.rng)
            redone_critic1_params, redone_opt_state = self.critic1_weight_recycler.maybe_update_weights(
                self.step, critic1_intermediates, new_critic.params, key, new_critic.opt_state_head
            )
            self.rng, key = jax.random.split(self.rng)
            redone_critic2_params, _ = self.critic2_weight_recycler.maybe_update_weights(
                self.step, critic2_intermediates, new_critic.params, key, new_critic.opt_state_head
            )
            new_critic_params = new_critic.params.copy(
                    add_or_replace={'CriticHead': 
                                    flax.core.FrozenDict({'critic0': redone_critic1_params['CriticHead']['critic0'], 
                                    'critic1': redone_critic2_params['CriticHead']['critic1']})})
            new_critic = new_critic.replace(params=new_critic_params,
                                            opt_state_head=redone_opt_state)
            
        if self.redo_actor:
            self.rng, key = jax.random.split(self.rng)
            redone_actor_params, redone_opt_state = self.actor_weight_recycler.maybe_update_weights(
                self.step, actor_intermediates, new_actor.params, key, new_actor.opt_state
            )
            new_actor = new_actor.replace(params=redone_actor_params,
                                          opt_state=redone_opt_state)

        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic
        self.temp = new_temp

        return info
