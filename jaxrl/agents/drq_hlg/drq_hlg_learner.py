"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax

from jaxrl.agents.drq.augmentations import batched_random_crop
from jaxrl.agents.drq.networks import ActivationTrackDrQDistributionalDoubleCritic, DrQDistributionalSingleCritic, DrQPolicy
from jaxrl.agents.sac import temperature
from jaxrl.agents.sac.critic import target_update
from jaxrl.datasets import Batch
from jaxrl.networks import policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey, ModelDecoupleOpt
from jaxrl.agents.drq_hlg import weight_recyclers
from jaxrl.utils import schedule


# MIN_VALUE = 0
# MAX_VALUE = 100 # 1+0.99+0.99**2+...+0.99**1000=100

@functools.partial(jax.jit, 
                   static_argnames=('update_target', 'n_logits', 'sigma', 'batch_size', 'double_q', \
                                    'use_entropy', 'probs_MSE', 'value_MSE'))
def _update_jit(
    n_logits: int, sigma: float, batch_size: int, double_q: bool, use_entropy: bool,
    min_value: float, max_value: float,
    rng: PRNGKey, actor: Model, critic: Model, target_critic: Model,
    temp: Model, batch: Batch, discount: float, tau: float,
    target_entropy: float, update_target: bool, probs_MSE: bool, value_MSE: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, InfoDict]:

    rng, key = jax.random.split(rng)

    support = jnp.linspace(min_value, max_value, n_logits + 1, dtype=jnp.float32) # logits are centers! (ie, num of classes)
    centers = (support[:-1] + support[1:]) / 2
    support = support[None, :].repeat(batch_size, axis=0) # (B, n_logits+1)
    
    def transform_to_probs(target): # (B,)
        target = jnp.clip(target, min_value, max_value)
        # print(target.shape, support.shape) # (512) (B, n_logits+1)
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
                                            soft_critic=True,
                                            probs_MSE=probs_MSE,
                                            value_MSE=value_MSE)
    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    # Use critic conv layers in actor:
    new_actor_params = actor.params.copy(
        add_or_replace={'SharedEncoder': new_critic.params['SharedEncoder']})
    # new_actor_batch_stats = actor.batch_stats.copy(
    #     add_or_replace={'batch_stats': new_critic.batch_stats})
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
                 track: bool,
                 replay_buffer,
                 redo_critic: bool,
                 redo_actor: bool,
                 neutralize_dormant_neurons: bool,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 reset_interval: int,
                 reset_start_step: int,
                 ntrlize_shared_dense: bool = False,
                 b1: float = 0.9,
                 b2: float = 0.999,
                 adam_eps: float = 1e-8,
                 use_LN_in_critic: bool = False,
                 use_WD_in_critic: bool = False,
                 use_LNWD_in_critic: bool = False,
                 WD_rate: float = 0.0001,
                 reset_mass_opt_state: bool = False,
                 ntrlize_thres: float = 2.,
                 NO_K_mass_thres: bool = True,
                 weight_scaling: bool = False,
                 incoming_scale: float = 10.0,
                 K: int = 5,
                 mass_thres: float = 10.,
                 dead_thres: float = 0.1,
                 weight_revive_eps: float = 0.01,
                 probs_MSE: bool = False,
                 value_MSE: bool = False,
                 use_layer_norm_in_critic: bool = False,
                 use_batch_norm: bool = False,
                 use_weight_decay_in_critic: bool = False,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 n_logits: int = 51,
                 sigma: float=1.5,
                 min_value: float = 0.,
                 max_value: float = 100.,
                 max_value_schedule: str = 'linear(10,100,500000)',
                 batch_size: int=256,
                 batch_size_statistics: int = 256,
                 dead_neurons_thresholds: Sequence[float] = [0., 0.025, 0.1],
                 dormancy_logging_period: int = 2_000,
                 double_q: bool = True,
                 use_entropy: bool = True,
                 actor_hidden_dims: Sequence[int] = (256, 256),
                 critic_hidden_dims: Sequence[int] = (256, 256),
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

        actor_def = DrQPolicy(actor_hidden_dims, action_dim, cnn_features,
                              cnn_strides, cnn_padding, latent_dim, use_batch_norm=use_batch_norm)
        
        actor = Model.create(actor_def,
                            inputs=[actor_key, observations],
                            tx=optax.adam(learning_rate=actor_lr))

        if double_q:
            critic_def = ActivationTrackDrQDistributionalDoubleCritic(critic_hidden_dims, n_logits, cnn_features, 
                                                                    cnn_strides, cnn_padding, latent_dim, 
                                                                    use_layer_norm=use_layer_norm_in_critic,
                                                                    use_batch_norm=use_batch_norm)
        else:
            critic_def = ActivationTrackDrQDistributionalDoubleCritic(critic_hidden_dims, n_logits, cnn_features, 
                                                    cnn_strides, cnn_padding, latent_dim, num_qs=1)
        # critic_def = DrQDoubleCritic(hidden_dims, cnn_features, cnn_strides,
        #                              cnn_padding, latent_dim)
        # critic = Model.create(critic_def,
        #                       inputs=[critic_key, observations, actions],
        #                       tx=optax.adam(learning_rate=critic_lr))
        if use_weight_decay_in_critic:
            optimizer = optax.adamw(learning_rate=critic_lr, weight_decay=WD_rate)
        else:
            optimizer = optax.adam(learning_rate=critic_lr)
        
        critic = ModelDecoupleOpt.create(critic_def,
                                        inputs=[critic_key, observations, actions],
                                        tx=optimizer,
                                        tx_enc=optimizer)
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
        self.use_batch_norm = use_batch_norm
        self.probs_MSE = probs_MSE
        self.value_MSE = value_MSE

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
            layer_list = [l for l in layer_list if ('dense' in l or 'final' in l)]
            print('layer name list: ', layer_list)
            return layer_list

        critic_layer_list = get_layer_list(critic)
        actor_layer_list = get_layer_list(actor)
        if ntrlize_shared_dense:
            critic1_layer_list = [l for l in critic_layer_list if 'critic0' in l or 'dense-1' in l]
        else:
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

        self.actor = actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.rng = rng
        self.step = 0
        self.redo_critic = redo_critic
        self.redo_actor = redo_actor
        self.ntrlize_shared_dense = ntrlize_shared_dense

        self.schedule = functools.partial(schedule, schdl=max_value_schedule)

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        batch_stats = self.actor.batch_stats if self.use_batch_norm else None
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature, use_batch_norm=self.use_batch_norm,
                                               batch_stats=batch_stats)

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

        max_value = self.schedule(step=self.step)
        self.max_value = max_value

        new_rng, new_actor, new_critic, new_target_critic, new_temp, info = _update_jit(
            self.n_logits, self.sigma, self.batch_size, self.double_q, self.use_entropy,
            self.min_value, self.max_value,
            self.rng, self.actor, self.critic, self.target_critic, self.temp,
            batch, self.discount, self.tau, self.target_entropy,
            self.step % self.target_update_period == 0, self.probs_MSE, self.value_MSE)
        
        # is_intermediated = self.critic1_weight_recycler.is_intermediated_required(self.step)
        # critic_intermediates, critic_preacts = (
        #     self.get_critic_intermediates(new_critic, new_critic.params) if is_intermediated else (None, None)
        # )
        # if is_intermediated:
        #     critic1_intermediates = {k: v for k, v in critic_intermediates.items() if 'critic0' in k or 'dense-1' in k}
        #     critic1_preacts = {k: v for k, v in critic_preacts.items() if 'critic0' in k or 'dense-1' in k}
        #     critic2_intermediates = {k: v for k, v in critic_intermediates.items() if 'critic1' in k}
        #     critic2_preacts = {k: v for k, v in critic_preacts.items() if 'critic1' in k}
        # else:
        #     critic1_intermediates, critic2_intermediates, critic1_preacts, critic2_preacts = [None] * 4
        # self.critic1_weight_recycler.maybe_log_deadneurons(
        #     self.step, critic1_intermediates, critic1_preacts, new_critic.params
        # ) # step-1: we log the first step's deadneurons
       
        is_intermediated = self.critic1_weight_recycler.is_intermediated_required(self.step)
        critic_intermediates, critic_preacts = (
            self.get_critic_intermediates(new_critic, new_critic.params) if is_intermediated else (None, None)
        )
        if is_intermediated:
            if self.ntrlize_shared_dense:
                critic1_intermediates = {k: v for k, v in critic_intermediates.items() if 'critic0' in k or 'dense-1' in k}
                critic1_preacts = {k: v for k, v in critic_preacts.items() if 'critic0' in k or 'dense-1' in k}
            else:
                critic1_intermediates = {k: v for k, v in critic_intermediates.items() if 'critic0' in k}
                critic1_preacts = {k: v for k, v in critic_preacts.items() if 'critic0' in k}
            critic2_intermediates = {k: v for k, v in critic_intermediates.items() if 'critic1' in k}
            critic2_preacts = {k: v for k, v in critic_preacts.items() if 'critic1' in k}
        else:
            critic1_intermediates, critic2_intermediates, critic1_preacts, critic2_preacts = [None] * 4

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
                                            'critic1': redone_critic2_params['CriticHead']['critic1']}),
                                    'dense-1_layernorm_tanh': 
                                        flax.core.FrozenDict(redone_critic1_params['dense-1_layernorm_tanh'])
                                    },
            )
            new_critic = new_critic.replace(params=new_critic_params,
                                            opt_state_head=redone_opt_state)
        self.critic1_weight_recycler.maybe_log_deadneurons(
            self.step, critic1_intermediates, critic1_preacts, new_critic.params
        )
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
