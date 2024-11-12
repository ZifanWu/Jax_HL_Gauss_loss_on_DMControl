import functools
import logging
import flax
from flax import linen as nn
import jax
from jax import random
import jax.numpy as jnp
import optax
import wandb


class BaseRecycler:
  """Base class for weight update methods.

  Attributes:
    all_layers_names: list of layer names in a model.
    recycle_type: neuron, layer based.
    dead_neurons_threshold: below this threshold a neuron is considered dead.
    reset_layers: list of layer names to be recycled.
    reset_start_layer_idx: index of the layer from which we start recycling.
    reset_period: int represents the period of weight update.
    reset_start_step: start recycle from start step
    reset_end_step:  end recycle from end step
    dormancy_logging_period:  the period of statistics logging e.g., dead neurons.
    prev_neuron_score: score at last reset step or log step in case of no reset.
    sub_mean_score: if True the average activation will be subtracted for each
      neuron when we calculate the score.
  """

  def __init__(
      self,
      all_layers_names,
      track,
      dead_neurons_threshold=0.1,
      reset_start_layer_idx=0,
      reset_period=200_000,
      reset_start_step=0,
      reset_end_step=100_000_000,
      dormancy_logging_period=20_000,
      sub_mean_score=False,
  ):
    self.all_layers_names = all_layers_names
    self.track = track
    self.dead_neurons_threshold = dead_neurons_threshold
    self.reset_layers = all_layers_names[reset_start_layer_idx:]
    self.reset_period = reset_period
    self.reset_start_step = reset_start_step
    self.reset_end_step = reset_end_step
    self.dormancy_logging_period = dormancy_logging_period
    self.prev_neuron_score = None
    self.sub_mean_score = sub_mean_score

    # NOTE (ZW) added
    self.historical_dormant_mask = None

  def update_reset_layers(self, reset_start_layer_idx):
    self.reset_layers = self.all_layers_names[reset_start_layer_idx:]

  def is_update_iter(self, step):
    return step > 0 and (step % self.reset_period == 0)

  def update_weights(self, intermediates, params, key, opt_state):
    raise NotImplementedError

  def maybe_update_weights(
      self, update_step, intermediates, params, key, opt_state
  ):
    self._last_update_step = update_step
    if self.is_reset(update_step):
      new_params, new_opt_state = self.update_weights(
          intermediates, params, key, opt_state
      )
    else:
      new_params, new_opt_state = params, opt_state
    return new_params, new_opt_state

  def is_reset(self, update_step):
    del update_step
    return False

  def is_intermediated_required(self, update_step):
    return self.is_logging_step(update_step)

  def is_logging_step(self, step):
    return step % self.dormancy_logging_period == 0

  def maybe_log_deadneurons(self, update_step, intermediates):
    is_logging = self.is_logging_step(update_step) # TODO debugging
    if is_logging:
      self.log_historical_dead_neuron_overlapping(intermediates, update_step)
      self.log_dead_neurons_count(intermediates, update_step)
  
  def log_historical_dead_neuron_overlapping(self, intermediates, update_step):
    """Track the overlapping rate of dead neurons between the historical set/and the current step.

    Args:
      intermediates: current intermediates

    Returns:
      log_dict: dict contains the percentage of intersection
    """
    score_tree = jax.tree_util.tree_map(self.estimate_neuron_score, intermediates)
    neuron_score_dict = flax.traverse_util.flatten_dict(score_tree, sep='/')

    if self.historical_dormant_mask is None:
      self.prev_neuron_score = neuron_score_dict
      log_dict = None
      self.historical_dormant_mask = {} # recording neurons that have at least once been detected dormant (whose entries take True)
      self.dormant_times = {} # recording the times each neuron is detected dormant
      # self.degree_of_dormancy = {} # recording (dormant_times) / (logging_times)
      self.n_log_historical_overlap = 1
    else:
      self.n_log_historical_overlap += 1
      log_dict = {}
      for prev_k_score, current_k_score in zip(
          self.prev_neuron_score.items(), neuron_score_dict.items()
      ): # layer k
        # print(prev_k_score[0], prev_k_score[1][0].shape) # Conv_0_act/__call__ (32,)
        _, prev_score = prev_k_score
        k, score = current_k_score
        prev_score, score = prev_score[0], score[0]
        prev_mask = prev_score <= self.dead_neurons_threshold
        prev_count = jnp.count_nonzero(prev_mask)
        nondead_mask = score > self.dead_neurons_threshold
        # we count the dead neurons which remains dead in the current step.
        curr_mask = score <= self.dead_neurons_threshold
        prev_intersect = curr_mask & prev_mask
        prev_intersect_count = jnp.count_nonzero(prev_intersect)

        if k not in self.historical_dormant_mask.keys(): # first log
          self.historical_dormant_mask[k] = prev_mask # non-dormant entries: False
          self.dormant_times[k] = jnp.zeros_like(prev_mask).astype(float)
          # self.degree_of_dormancy[k] = jnp.zeros_like(curr_mask).astype(float)
        self.dormant_times[k] += curr_mask.astype(float)
        # TODO (ZW) what if we reset a neuron only if its degree_of_dormancy has reached a threshold
        degree_of_dormancy = self.dormant_times[k] / self.n_log_historical_overlap
        # avg_degree_of_dormancy = degree_of_dormancy.mean()

        pre_hist_dead_count = jnp.count_nonzero(self.historical_dormant_mask[k])
        self.historical_dormant_mask[k] = (self.historical_dormant_mask[k]) | (curr_mask) # NOTE (ZW) merging the current dormant set into the historical set

        intersected_mask = (self.historical_dormant_mask[k]) & (curr_mask)
        intersected_count = jnp.count_nonzero(intersected_mask)
        curr_dead_count = jnp.count_nonzero(curr_mask)
        # hist_dead_count = jnp.count_nonzero(self.historical_dormant_mask[k])
        post_hist_dead_count = jnp.count_nonzero(self.historical_dormant_mask[k])
        denominator = post_hist_dead_count # This implements the post-merging-set-as-denominator metric
        # denominator = max(curr_dead_count, hist_dead_count) # This implements the max-pre-merging-set-as-denominator metric

        # self.historical_dormant_mask[k] = (self.historical_dormant_mask[k]) | (curr_mask)
        percent = (
            (float(intersected_count) / denominator.item())
            if denominator
            else 0.0
        )
        prev_intersect_percent = (
          (float(prev_intersect_count) / prev_count)
          if prev_count
          else 0.0
        )

        if self.track:
          wandb.log({'{}_historical_overlap_rate'.format(k[14:-13]): percent, 'grad_step': update_step})
          wandb.log({'{}_current_historical_ratio(pre_merging)'.format(k[14:-13]): (curr_dead_count / pre_hist_dead_count).item(), 'grad_step': update_step})
          wandb.log({'{}_historical_dormant_count(post_merging)'.format(k[14:-13]): post_hist_dead_count.item(), 'grad_step': update_step})
          wandb.log({'{}_mean_score_recycled'.format(k[14:-13]): jnp.mean(score[prev_mask]), 'grad_step': update_step})
          wandb.log({'{}_mean_score_nondead'.format(k[14:-13]): jnp.mean(score[nondead_mask]), 'grad_step': update_step})
          wandb.log({'{}_intersected_rate'.format(k[14:-13]): prev_intersect_percent, 'grad_step': update_step})
      self.prev_neuron_score = neuron_score_dict
    return log_dict

  def _score2mask(self, activation, param, next_param, key):
    del key, param, next_param
    score = self.estimate_neuron_score(activation)
    return score <= self.dead_neurons_threshold

  def log_outgoing_weights_magnitude(self, param_dict, activations_dict, key):
    """log: # TODO (ZW)
    1. magnitude of dormant neurons's outgoing_weights (in each layer)
    2. magnitude of the rest neurons' outgoing_weights in the same layer
    """
    incoming_mask_dict = {
        k: jnp.zeros_like(p) if p.ndim != 1 else None
        for k, p in param_dict.items()
    }
    outgoing_mask_dict = {
        k: jnp.zeros_like(p) if p.ndim != 1 else None
        for k, p in param_dict.items()
    }
    ingoing_random_keys_dict = {k: None for k in param_dict}
    outgoing_random_keys_dict = (
        {k: None for k in param_dict}
        if self.init_method_outgoing == 'random'
        else {}
    )

    # prepare mask of incoming and outgoing recycled connections
    for k in self.reset_layers:
      param_key = 'params/' + k + '/kernel'
      param = param_dict[param_key]
      # This won't work for DRQ, since returned keys can be a list.
      # We don't support that at the moment.
      next_key = self.next_layers[k]
      if isinstance(next_key, list):
        next_key = next_key[0]
      next_param = param_dict['params/' + next_key + '/kernel']
      activation = activations_dict[k + '_act/__call__'][0]
      neuron_mask = self._score2mask(activation, param, next_param, key)

      # the for loop handles the case where a layer has multiple next layers
      # like the case in DrQ where the output layer has multihead.
      next_keys = (
          self.next_layers[k]
          if isinstance(self.next_layers[k], list)
          else [self.next_layers[k]]
      )
      for next_k in next_keys:
        next_param_key = 'params/' + next_k + '/kernel'
        next_param = param_dict[next_param_key]
        incoming_mask, outgoing_mask = self.create_mask_helper(
            neuron_mask, param, next_param
        )
        incoming_mask_dict[param_key] = incoming_mask
        outgoing_mask_dict[next_param_key] = outgoing_mask
        key, subkey = random.split(key)
        ingoing_random_keys_dict[param_key] = subkey
        if self.init_method_outgoing == 'random':
          key, subkey = random.split(key)
          outgoing_random_keys_dict[next_param_key] = subkey

    #     if self.prune_dormant_neurons: # NOTE (ZW) stop the gradients flowing through dormant neurons
    #       # NOTE (ZW) Log the magnitude of outgoing weights of dormant neurons
    #       print('Pruning {} outgoing weights at layer {}'.format(outgoing_mask.sum(), k))
    #       next_param = jnp.where(~outgoing_mask, next_param, jax.lax.stop_gradient(next_param))
        
    #       if self.first_time_pruning and (jnp.count_nonzero(outgoing_mask) > 0):
    #         self.next_param_key = next_param_key
    #         self.outgoing_mask = outgoing_mask
    #         self.first_time_pruning = False

    # if (not self.first_time_pruning) and self.prune_dormant_neurons:
    #   print(jnp.count_nonzero(self.outgoing_mask), self.next_param_key)
    #   frozen_params = jnp.where(self.outgoing_mask == 1, jnp.zeros_like(param_dict[self.next_param_key]), param_dict[self.next_param_key])
    #   print(jnp.linalg.vector_norm(frozen_params).item())
    #   import time
    #   time.sleep(2)
    #   if self.track:
    #     wandb.log({'frozen_params_norm': jnp.linalg.vector_norm(frozen_params).item(), 'grad_step': self._last_update_step})

      # reset bias
      bias_key = 'params/' + k + '/bias'
      new_bias = jnp.zeros_like(param_dict[bias_key])
      param_dict[bias_key] = jnp.where(
          neuron_mask, new_bias, param_dict[bias_key]
      ) # True entities in param_dict[bias_key] will be replaced by new_bias

    return (
        incoming_mask_dict,
        outgoing_mask_dict,
        ingoing_random_keys_dict,
        outgoing_random_keys_dict,
        param_dict,
    )

  def log_dead_neurons_count(self, intermediates, update_step):
    """log dead neurons in each layer.

    For conv layer we also log dead elements in the spatial dimension.

    Args:
      intermediates: intermidate activation in each layer.

    Returns:
      log_dict_elements_per_neuron
      log_dict_neurons
    """

    def log_dict(score, score_type):
      total_neurons, total_deadneurons = 0.0, 0.0
      score_dict = flax.traverse_util.flatten_dict(score, sep='/')
    #   print(score_dict.keys()) #['reused_critic/critic0/dense0_act/__call__', 'reused_critic/critic0/dense1_act/__call__', 'reused_critic/critic1/dense0_act/__call__', 'reused_critic/critic1/dense1_act/__call__']

      log_dict = {}
      layer_count = 0
      for k, m in score_dict.items():
        layer_count += 1
        if 'final_layer' in k:
          continue
        m = m[0]
        layer_size = float(jnp.size(m))
        deadneurons_count = jnp.count_nonzero(m <= self.dead_neurons_threshold)
        total_neurons += layer_size
        total_deadneurons += deadneurons_count
        log_dict[f'dead_{score_type}_count/{k[:-9]}'] = float(deadneurons_count)
        # print('{} dormant neuron percentage'.format(k[14:-13]), float(deadneurons_count) / layer_size)
        if self.track:
          wandb.log({'{} dormant percentage'.format(k[14:-13]): 
                     float(deadneurons_count) / layer_size, 'grad_step': update_step})
      if self.track:
        wandb.log({'overall dormant percentage': 
                   float(total_deadneurons) / total_neurons, 'grad_step': update_step})
      return log_dict

    neuron_score = jax.tree_util.tree_map(self.estimate_neuron_score, intermediates)
    log_dict_neurons = log_dict(neuron_score, 'feature')

    return log_dict_neurons

  def estimate_neuron_score(self, activation, is_cbp=False):
    """Calculates neuron score based on absolute value of activation.

    The score of feature map is the normalized average score over
    the spatial dimension.

    Args:
      activation: intermediate activation of each layer
      is_cbp: if true, subtracts the mean and skips normalization.

    Returns:
      element_score: score of each element in feature map in the spatial dim.
      neuron_score: score of feature map
    """
    reduce_axes = list(range(activation.ndim - 1))
    if self.sub_mean_score or is_cbp:
      activation = activation - jnp.mean(activation, axis=reduce_axes)

    score = jnp.mean(jnp.abs(activation), axis=reduce_axes)
    if not is_cbp:
      # Normalize so that all scores sum to one.
      score /= jnp.mean(score) + 1e-9

    return score



