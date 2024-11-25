"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple
import re
from typing import Tuple

import flax.traverse_util
from optax._src import base
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
import flax.linen as nn

from jaxrl.agents.drq_v2.augmentations import batched_random_crop, drqv2_augmentation
from jaxrl.agents.drq_v2.networks import DrQv2DoubleCritic, DrQv2Policy, DrQv2Policy, Encoder
from jaxrl.agents.drq_v2.actor import update as update_actor
from jaxrl.agents.drq_v2.critic import target_update
from jaxrl.agents.drq_v2.critic import update as update_critic
from jaxrl.datasets import Batch
from jaxrl.networks import policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey, ModelDecoupleOpt
from jaxrl.agents.drq_v2 import weight_recyclers
from jaxrl.utils import schedule


@functools.partial(jax.jit, static_argnames=('update_target', 'use_batched_random_crop'))
def _update_jit(
    rng: PRNGKey, stddev: float, encoder: Model, actor: Model, critic: Model, target_critic: Model,
    use_batched_random_crop: bool, stddev_clip: float, batch: Batch, discount: float, tau: float,
    target_entropy: float, update_target: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)
    # print(batch.observations) (256, 84, 84, 9)
    if use_batched_random_crop:
        observations = batched_random_crop(key, batch.observations)
    else:
        observations = drqv2_augmentation(batch.observations, key=key)
    rng, key = jax.random.split(rng)
    if use_batched_random_crop:
        next_observations = batched_random_crop(key, batch.next_observations)
    else:
        next_observations = drqv2_augmentation(batch.next_observations, key=key)

    batch = batch._replace(observations=observations,
                           next_observations=next_observations)

    rng, key = jax.random.split(rng)
    new_encoder, new_critic, critic_info = update_critic(key,
                                                         stddev,
                                                        encoder,
                                                        actor,
                                                        critic,
                                                        target_critic,
                                                        batch,
                                                        discount,
                                                        stddev_clip,
                                                        soft_critic=True)
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, stddev, new_encoder, stddev_clip, actor, new_critic, batch)

    return rng, new_encoder, new_actor, new_critic, new_target_critic, {
        **critic_info,
        **actor_info,
    }


class DrQv2Learner(object):

    def __init__(self,
                 seed: int,
                 track: bool,
                 replay_buffer,
                 redo: bool,
                 use_batched_random_crop: bool,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 reset_interval: int = 200_000,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 stddev_schedule: str = 'linear(1.0,0.1,500000)',
                 stddev_clip: float = 0.3,
                 hidden_dims: Sequence[int] = (256, 256),
                 batch_size: int = 512,
                 batch_size_statistics: int = 256,
                 dead_neurons_threshold: float = 0.1,
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
        # print(actions.shape, observations.shape) # cheetah-run: (1, 6) (1, 84, 84, 9) where 9 means 3 steps and 3 channels for each step's obs
        # import time
        # time.sleep(21)

        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, encoder_key = jax.random.split(rng, 4)

        encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        encoder = Model.create(encoder_def, 
                               inputs=[encoder_key, observations],
                               tx=optax.adam(learning_rate=critic_lr))

        actor_def = DrQv2Policy(hidden_dims, action_dim, latent_dim)
        actor = Model.create(actor_def,
                             inputs=[actor_key, np.zeros((1, 32*35*35))],
                             tx=optax.adam(learning_rate=actor_lr))

        critic_def = DrQv2DoubleCritic(hidden_dims, latent_dim)
        critic = Model.create(critic_def,
                              inputs=[critic_key, np.zeros((1, 32*35*35)), actions],
                              tx=optax.adam(learning_rate=critic_lr))
        target_critic = Model.create(
            critic_def, inputs=[critic_key, np.zeros((1, 32*35*35)), actions])

        self.encoder = encoder
        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.rng = rng
        self.step = 0

        def get_layer_list(model: Model) -> list[str]:
            param_dict = flax.traverse_util.flatten_dict(model.params, sep='/')
            layer_list = list(param_dict.keys())
            # print(1111, layer_list)
            layer_list = [l[l.find('/')+1:l.rfind('/')] for l in layer_list]
            # print(2222, layer_list)
            layer_list = list(dict.fromkeys(layer_list))
            # print(3333, layer_list)
            layer_list = [l for l in layer_list if 'final' not in l and l != '']
            print(4444, layer_list)
            return layer_list

        critic_layer_list = get_layer_list(critic)
        actor_layer_list = get_layer_list(actor)
        if redo:
            self.critic_weight_recycler = weight_recyclers.NeuronRecycler(critic_layer_list, 
                                                                          track, 
                                                                          dead_neurons_threshold=dead_neurons_threshold, 
                                                                          dormancy_logging_period=dormancy_logging_period, 
                                                                          prune_dormant_neurons=False, 
                                                                          reset_period=reset_interval)
            self.actor_weight_recycler = weight_recyclers.NeuronRecycler(actor_layer_list, 
                                                                          track, 
                                                                          dead_neurons_threshold=dead_neurons_threshold, 
                                                                          dormancy_logging_period=dormancy_logging_period, 
                                                                          prune_dormant_neurons=False, 
                                                                          reset_period=reset_interval)
        else:
            self.critic_weight_recycler = weight_recyclers.BaseRecycler(critic_layer_list, 
                                                                        track, 
                                                                        dead_neurons_threshold=dead_neurons_threshold, 
                                                                        dormancy_logging_period=dormancy_logging_period, 
                                                                        )
            self.actor_weight_recycler = weight_recyclers.BaseRecycler(actor_layer_list, 
                                                                        track, 
                                                                        dead_neurons_threshold=dead_neurons_threshold, 
                                                                        dormancy_logging_period=dormancy_logging_period, 
                                                                        )

        self.replay_buffer = replay_buffer
        self.batch_size_statistics = batch_size_statistics
        self.redo = redo

        self.use_batched_random_crop = use_batched_random_crop
        self.stddev_clip = stddev_clip
        self.schedule = functools.partial(schedule, schdl=stddev_schedule)

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0,
                       step = None) -> jnp.ndarray:
        stddev = self.schedule(step=step)
        encodings = self.encoder(observations)
        rng, actions = policies.sample_deterministic_actions(self.rng, stddev, self.actor.apply_fn,
                                                            self.actor.params, encodings, temperature)
        self.rng = rng

        return actions

    # def sample_actions(self,
    #                    observations: np.ndarray,
    #                    temperature: float = 1.0) -> jnp.ndarray: # DrQPolicy
    #     encodings = self.encoder(observations)
    #     rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
    #                                            self.actor.params, encodings,
    #                                            temperature)

    #     self.rng = rng

    #     actions = np.asarray(actions)
    #     return np.clip(actions, -1, 1)
    
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

    def update(self, batch: Batch, step = None) -> InfoDict:
        self.step += 1
        stddev = self.schedule(step=step)
        new_rng, new_encoder, new_actor, new_critic, new_target_critic, info = _update_jit(
            self.rng, stddev, self.encoder, self.actor, self.critic, self.target_critic,
            self.use_batched_random_crop, self.stddev_clip, batch, self.discount, self.tau, self.target_entropy,
            self.step % self.target_update_period == 0)
        
        is_intermediated = self.critic_weight_recycler.is_intermediated_required(self.step)
        intermediates = (
            self.get_intermediates(new_critic, new_critic.params) if is_intermediated else None
        )
        critic_intermediates = intermediates
        self.critic_weight_recycler.maybe_log_deadneurons(
            self.step, critic_intermediates
        )
        actor_intermediates = (
            self.get_intermediates(new_actor, new_actor.params) if is_intermediated else None
        )
        self.actor_weight_recycler.maybe_log_deadneurons(
            self.step, actor_intermediates
        )

        self.rng = new_rng
        if self.redo:
            self.rng, key = jax.random.split(self.rng)
            redone_critic_params, redone_critic_opt_state = self.critic_weight_recycler.maybe_update_weights(
                self.step, critic_intermediates, new_critic.params, key, [new_critic.opt_state_enc, new_critic.opt_state_head]
            )
            new_critic = new_critic.replace(params=redone_critic_params, 
                                            opt_state_enc=redone_critic_opt_state[0],
                                            opt_state_head=redone_critic_opt_state[1])
        self.encoder = new_encoder
        self.actor = new_actor
        self.critic = new_critic
        self.target_critic = new_target_critic

        return info
