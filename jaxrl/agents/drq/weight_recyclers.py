import functools
import logging
import flax
from flax import linen as nn
import jax
from jax import random
import jax.numpy as jnp
import jax.scipy.stats as jstats
import optax
import wandb


def leastk_mask(scores, ones_fraction):
  """Given a tensor of scores creates a binary mask.

  Args:
    scores: top-scores are kept
    ones_fraction: float, of the generated mask.

  Returns:
    array, same shape and type as scores or None.
  """
  if ones_fraction is None or ones_fraction == 0:
    return jnp.zeros_like(scores)
  # This is to ensure indices with smallest values are selected.
  scores = -scores

  n_ones = jnp.round(jnp.size(scores) * ones_fraction)
  k = jnp.maximum(1, n_ones).astype(int)
  flat_scores = jnp.reshape(scores, -1)
  threshold = jax.lax.sort(flat_scores)[-k]

  mask = (flat_scores >= threshold).astype(flat_scores.dtype)
  return mask.reshape(scores.shape)


def reset_momentum(momentum, mask):
  new_momentum = momentum if mask is None else momentum * (1.0 - mask)
  return new_momentum


def weight_reinit_zero(param, mask):
  if mask is None:
    return param
  else:
    new_param = jnp.zeros_like(param)
    param = jnp.where(mask == 1, new_param, param)
    return param
  

def weight_shrink(param, mask, k):
  param = jnp.where(mask == 1, param / k, param)
  return param
  

def weight_revive(param, next_param, dead_neuron_mask, key, 
                  mass_incoming_mask, mass_outgoing_mask,
                  K, M, eps, k):
  '''
    dead_mask: the incoming-weight mask of ONE dead neuron
  '''
  new_incoming_param = (param * mass_incoming_mask) / k
  new_outgoing_param = next_param * mass_outgoing_mask
  dead_incoming_mask, dead_outgoing_mask = create_mask_helper(
      dead_neuron_mask, param, next_param
  )
  key, subkey = random.split(key)
  noise = jax.random.normal(subkey) * eps
  param = jnp.where(
    dead_incoming_mask, new_incoming_param + noise, param
  )
  key, subkey = random.split(key)
  noise = jax.random.normal(subkey) * eps
  next_param = jnp.where(
    dead_outgoing_mask, new_outgoing_param + noise, next_param
  )
  return param, next_param, key


def create_mask_helper(neuron_mask, current_param, next_param):
  """generate incoming and outgoing weight mask given dead neurons mask.

  Args:
    neuron_mask: mask of size equals the width of a layer.
    current_param: incoming weights of a layer.
    next_param: outgoing weights of a layer.

  Returns:
    incoming_mask
    outgoing_mask
  """

  def mask_creator(expansion_axis, expansion_axes, param, neuron_mask):
    """create a mask of weight matrix given 1D vector of neurons mask.

    Args:
      expansion_axis: List contains 1 axis. The dimension to expand the mask
        for dense layers (weight shape 2D).
      expansion_axes: List conrtains 3 axes. The dimensions to expand the
        score for convolutional layers (weight shape 4D).
      param: weight.
      neuron_mask: 1D mask that represents dead neurons(features).

    Returns:
      mask: mask of weight.
    """
    axes = expansion_axis
    # flatten layer
    # The size of neuron_mask is the same as the width of last conv layer.
    # This conv layer will be flatten and connected to dense layer.
    # we repeat each value of a feature map to cover the spatial dimension.
    if axes[0] == 1 and (param.shape[0] > neuron_mask.shape[0]):
      num_repeatition = int(param.shape[0] / neuron_mask.shape[0])
      neuron_mask = jnp.repeat(neuron_mask, num_repeatition, axis=0)
    mask = jnp.expand_dims(neuron_mask, axis=tuple(axes))
    for i in range(len(axes)):
      mask = jnp.repeat(mask, param.shape[axes[i]], axis=axes[i])
    return mask

  incoming_mask = mask_creator([0], [0, 1, 2], current_param, neuron_mask)
  outgoing_mask = mask_creator([1], [0, 1, 3], next_param, neuron_mask)
  return incoming_mask, outgoing_mask


def weight_reinit_random(
    param, mask, key, weight_scaling=False, scale=1.0, weights_type='incoming'
):
  """Randomly reinit recycled weights and may scale its norm.

  If scaling applied, the norm of recycled weights equals
  the average norm of non recycled weights per neuron multiplied by a scalar.

  Args:
    param: current param
    mask: incoming/outgoing mask for recycled weights
    key: random key to generate new random weights
    weight_scaling: if true scale recycled weights with the norm of non recycled
    scale: scale to multiply the new weights norm.
    weights_type: incoming or outgoing weights

  Returns:
  params: new params after weight recycle.
  """
  if mask is None or key is None:
    return param

  new_param = nn.initializers.xavier_uniform()(key, shape=param.shape)

  if weight_scaling:
    axes = list(range(param.ndim))
    if weights_type == 'outgoing':
      del axes[-2]
    else:
      del axes[-1]

    neuron_mask = jnp.mean(mask, axis=axes)

    non_dead_count = neuron_mask.shape[0] - jnp.count_nonzero(neuron_mask)
    norm_per_neuron = _get_norm_per_neuron(param, axes)
    non_recycled_norm = (
        jnp.sum(norm_per_neuron * (1 - neuron_mask)) / non_dead_count
    )
    non_recycled_norm = non_recycled_norm * scale

    normalized_new_param = _weight_normalization_per_neuron_norm(
        new_param, axes
    )
    new_param = normalized_new_param * non_recycled_norm

  param = jnp.where(mask == 1, new_param, param)
  return param


def _weight_normalization_per_neuron_norm(param, axes):
  norm_per_neuron = _get_norm_per_neuron(param, axes)
  norm_per_neuron = jnp.expand_dims(norm_per_neuron, axis=axes)
  normalized_param = param / norm_per_neuron
  return normalized_param


def _get_norm_per_neuron(param, axes):
  return jnp.sqrt(jnp.sum(jnp.power(param, 2), axis=axes))


@jax.jit
def compute_quantiles(arr: jnp.ndarray, q):
  return jnp.quantile(arr, q)


@jax.jit
def sort_array(arr: jnp.ndarray):
  # Sort the array in descending order and return the indices
  return jnp.argsort(arr)[::-1]


def topK_and_leastKM_elements(arr: jnp.ndarray, K: int):
  indices = sort_array(arr)
  top_K_indices = indices[:K]
  top_K_values = arr[top_K_indices]
  M = top_K_values[-1].astype(int)
  least_KM_indices = indices[-K * M:]
  least_KM_values = arr[least_KM_indices]
  return top_K_values, top_K_indices, least_KM_values, least_KM_indices, M


@jax.jit
def check_normality(data: jnp.ndarray):
  mean, std = jnp.mean(data, axis=1, keepdims=True), jnp.std(data, axis=1, keepdims=True) # mean and std for each feature vector in the batch
  empirical_quantiles = []
  theoretical_quantiles = []
  for i in range(9):
    empirical_quantiles.append(jnp.mean(data <= mean + (i-4) * 0.5 * std, axis=1))
    theoretical_quantiles.append([jstats.norm.cdf((i-4)*0.5, 0, 1)])
  return jnp.sum(jnp.abs(jnp.array(empirical_quantiles) - jnp.array(theoretical_quantiles)))


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
      dead_neurons_thresholds=[0, 0.025, 0.1],
      reset_start_layer_idx=0,
      reset_period=200_000,
      reset_start_step=0,
      reset_end_step=100_000_000,
      dormancy_logging_period=20_000,
      sub_mean_score=False,
  ):
    self.all_layers_names = all_layers_names
    self.track = track
    self.dead_neurons_thresholds = dead_neurons_thresholds
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
    return self.is_logging_step(update_step) # TODO debugging

  def is_logging_step(self, step):
    return step % self.dormancy_logging_period == 0

  def maybe_log_deadneurons(self, update_step, intermediates, preactivations, params):
    is_logging = self.is_logging_step(update_step)
    if is_logging: # TODO debugging
      self.log_historical_dead_neuron_overlapping(intermediates, preactivations, params, update_step)
  
  def _compute_mask(self, score_dict):
    masks = []
    for threshold in self.dead_neurons_thresholds:
      masks.append(score_dict <= threshold)
    return masks
      
  def _compute_nondead_mask(self, score_dict):
    masks = []
    for threshold in self.dead_neurons_thresholds:
      masks.append(score_dict > threshold)
    return masks
  
  def log_historical_dead_neuron_overlapping(self, intermediates, preactivations, params, update_step):
    """Track the overlapping rate of dead neurons between the historical set/and the current step.

    Args:
      intermediates: current intermediates

    Returns:
      log_dict: dict contains the percentage of intersection
    """
    from collections import OrderedDict
    intermediates = OrderedDict(intermediates)
    score_tree = jax.tree_util.tree_map(self.estimate_neuron_score, intermediates)
    neuron_score_dict = flax.traverse_util.flatten_dict(score_tree, sep='/')
    activation_dict = flax.traverse_util.flatten_dict(intermediates, sep='/')
    preactivation_dict = flax.traverse_util.flatten_dict(preactivations, sep='/')
    param_dict = flax.traverse_util.flatten_dict(params, sep='/')

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
      dense0_dormancy_masks = []
      dense_top3_indices = []
      total_dead_count, total_neurons = 0, 0
      for prev_k_score, current_k_score, activation_k, preactivation_k in zip(
          self.prev_neuron_score.items(), neuron_score_dict.items(), 
          activation_dict.items(), preactivation_dict.items()
      ): # layer k
        # print(prev_k_score[0], prev_k_score[1][0].shape) # Conv_0_act/__call__ (32,)
        _, prev_score = prev_k_score
        k, score = current_k_score
        _, activation = activation_k
        k_pre, preactivation = preactivation_k
        # print(activation_dict[k][0].shape, prev_score[0].shape, score[0].shape)(256, 21, 21, 32) (32,) (32,)
        prev_score, score, activation, preactivation = prev_score[0], score[0], \
                                                       activation[0], preactivation[0]
        reduce_axes = list(range(activation.ndim - 1)) # more than 2 dims when it's a CNN
        activation = jnp.mean(jnp.abs(activation), axis=reduce_axes)
        # preactivation = jnp.mean(preactivation, axis=reduce_axes)
        prev_masks = self._compute_mask(prev_score)
        # we count the dead neurons which remains dead in the current step.
        curr_masks = self._compute_mask(score)
        curr_nondead_masks = self._compute_nondead_mask(score)
        # import flax.linen as nn
        # print(jnp.count_nonzero(nn.relu(preactivation) == activation), jnp.size(activation), jnp.count_nonzero(nn.relu(preactivation) == activation) == jnp.count_nonzero(jnp.maximum(preactivation, 0)==activation))
        # print(jnp.max(jnp.abs(nn.relu(preactivation) - activation)))
        # print(jnp.count_nonzero(score<=0), jnp.count_nonzero(activation==0), score.shape, activation.shape)

        layer_name = k[k.find('/')+1:k.rfind('/')-4]
        if self.track and 'dense' in k and ('critic0' in k or 'actor' in k):
          wandb.log({'{}_mean_activation'.format(layer_name): jnp.mean(activation), 'grad_step': update_step})
          wandb.log({'{}_mean_preactivation'.format(layer_name): jnp.mean(preactivation), 'grad_step': update_step})
          # wandb.log({'{}_std_preactivation'.format(layer_name): jnp.std(preactivation), 'grad_step': update_step})

          # we want to check if the preactivation distribution within a layer is some Gaussian, 
          # so we check if quantiles matches the theoretical quantiles of the Gaussian with that mean and that std
          cdf_diff = check_normality(preactivation)
          wandb.log({'{}_cdf_difference'.format(layer_name): cdf_diff, 'grad_step': update_step})
          # q = jnp.array([0.25, 0.5, 0.75])
          # quantiles = jnp.quantile(jnp.mean(preactivation, axis=0), q) # (q.shape)
          quantiles = compute_quantiles(jnp.mean(preactivation, axis=0), jnp.array([0.25, 0.5, 0.75]))
          wandb.log({'{}_preact_1qt'.format(layer_name): quantiles[0], 'grad_step': update_step})
          wandb.log({'{}_preact_2qt'.format(layer_name): quantiles[1], 'grad_step': update_step})
          wandb.log({'{}_preact_3qt'.format(layer_name): quantiles[2], 'grad_step': update_step})
        thres_idx = 0
        for curr_mask, prev_mask, curr_nondead_mask in zip(curr_masks, prev_masks, curr_nondead_masks):
          if ('critic0' in k or 'actor' in k) and 'dense0' in k:
            dense0_dormancy_masks.append(curr_mask.copy())
          prev_intersect = curr_mask & prev_mask
          prev_intersect_count = jnp.count_nonzero(prev_intersect)
          prev_count = jnp.count_nonzero(prev_mask)

          if k not in self.historical_dormant_mask.keys(): # first log
            self.historical_dormant_mask[k] = prev_mask # non-dormant entries: False

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

          if self.track and 'dense' in k and ('critic0' in k or 'actor' in k):
            wandb.log({'{}_{}_historical_overlap_rate'.format(layer_name, self.dead_neurons_thresholds[thres_idx]): percent, 'grad_step': update_step})
            wandb.log({'{}_{}_current_historical_ratio(pre_merging)'.format(layer_name, self.dead_neurons_thresholds[thres_idx]): (curr_dead_count / pre_hist_dead_count).item(), 'grad_step': update_step})
            wandb.log({'{}_{}_historical_dormant_count(post_merging)'.format(layer_name, self.dead_neurons_thresholds[thres_idx]): post_hist_dead_count.item(), 'grad_step': update_step})
            wandb.log({'{}_{}_dead_intersected_percent'.format(layer_name, self.dead_neurons_thresholds[thres_idx]): prev_intersect_percent, 'grad_step': update_step})
            wandb.log({'{}_{}_dormant_percentage'.format(layer_name, self.dead_neurons_thresholds[thres_idx]): float(curr_dead_count) / jnp.size(score), 'grad_step': update_step})

            wandb.log({'{}_{}_mean_activation_recycled'.format(layer_name, self.dead_neurons_thresholds[thres_idx]): jnp.mean(activation[prev_mask]), 'grad_step': update_step})
            wandb.log({'{}_{}_mean_activation_nondead'.format(layer_name, self.dead_neurons_thresholds[thres_idx]): jnp.mean(activation[curr_nondead_mask]), 'grad_step': update_step})
            wandb.log({'{}_{}_mean_activation_dead'.format(layer_name, self.dead_neurons_thresholds[thres_idx]): jnp.mean(activation[curr_mask]), 'grad_step': update_step})

            wandb.log({'{}_{}_mean_preactivation_recycled'.format(layer_name, self.dead_neurons_thresholds[thres_idx]): jnp.mean(preactivation[:, prev_mask]), 'grad_step': update_step})
            wandb.log({'{}_{}_mean_preactivation_nondead'.format(layer_name, self.dead_neurons_thresholds[thres_idx]): jnp.mean(preactivation[:, curr_nondead_mask]), 'grad_step': update_step})
            wandb.log({'{}_{}_mean_preactivation_dead'.format(layer_name, self.dead_neurons_thresholds[thres_idx]): jnp.mean(preactivation[:, curr_mask]), 'grad_step': update_step})
          thres_idx += 1
        # log top activations
        if self.track and 'dense' in k and ('critic0' in k or 'actor' in k):
          top3_values, top3_indices, _, _, M = topK_and_leastKM_elements(activation, 3)
          dense_top3_indices.append(top3_indices)
          wandb.log({'{}_top1_activation'.format(layer_name): top3_values[0], 'grad_step': update_step})
          wandb.log({'{}_top2_activation'.format(layer_name): top3_values[1], 'grad_step': update_step})
          wandb.log({'{}_top3_activation'.format(layer_name): top3_values[2], 'grad_step': update_step})
          wandb.log({'{}_top1_idx'.format(layer_name): top3_indices[0], 'grad_step': update_step})
          wandb.log({'{}_top2_idx'.format(layer_name): top3_indices[1], 'grad_step': update_step})
          wandb.log({'{}_top3_idx'.format(layer_name): top3_indices[2], 'grad_step': update_step})
          sum_acti = jnp.sum(activation)
          wandb.log({'{}_mean_excluding_top1_activation'.format(layer_name): (sum_acti-top3_values[0])/(jnp.size(activation)-1), 'grad_step': update_step})
          wandb.log({'{}_mean_excluding_top2_activation'.format(layer_name): (sum_acti-top3_values[0]-top3_values[1])/(jnp.size(activation)-2), 'grad_step': update_step})
          wandb.log({'{}_mean_excluding_top3_activation'.format(layer_name): (sum_acti-top3_values.sum())/(jnp.size(activation)-3), 'grad_step': update_step})

      # log weights
      if self.track and self.historical_dormant_mask is not None:
        for k in self.reset_layers:
          if 'dense' in k and ('critic0' in k or 'actor' in k):
            top3_indices = dense_top3_indices.pop(0)
            param_key = k + '/kernel'
            bias_key = k + '/bias'
            abs_param = jnp.abs(param_dict[param_key]) # (8, 8, 4, 32)(4, 4, 32, 64)(3, 3, 64, 64)(7744, 512)(512, 6)
            wandb.log({'{}_w_mean'.format(k): abs_param.mean(), 'grad_step': update_step})
            quantiles = compute_quantiles(abs_param, jnp.array([0.25, 0.5, 0.75]))
            wandb.log({'{}_w_1qt'.format(k): quantiles[0], 'grad_step': update_step})
            wandb.log({'{}_w_2qt'.format(k): quantiles[1], 'grad_step': update_step})
            wandb.log({'{}_w_3qt'.format(k): quantiles[2], 'grad_step': update_step})
            wandb.log({'{}_top1_out_w_mean'.format(k): jnp.mean(abs_param[top3_indices[0]]), 'grad_step': update_step})
            wandb.log({'{}_top2_out_w_mean'.format(k): jnp.mean(abs_param[top3_indices[1]]), 'grad_step': update_step})
            wandb.log({'{}_top3_out_w_mean'.format(k): jnp.mean(abs_param[top3_indices[2]]), 'grad_step': update_step})
            if ('critic0' in k or 'actor' in k) and 'dense1' in k:
              for idx, dormancy_mask in enumerate(dense0_dormancy_masks):
                wandb.log({'Dense_0_{}_dormant_out_w_mean'.format(self.dead_neurons_thresholds[idx]): jnp.mean(abs_param[dormancy_mask].mean()), 'grad_step': update_step})
            
            abs_bias = jnp.abs(param_dict[bias_key])
            wandb.log({'{}_b_mean'.format(k): abs_bias.mean(), 'grad_step': update_step})
            quantiles = compute_quantiles(abs_bias, jnp.array([0.25, 0.5, 0.75]))
            wandb.log({'{}_b_1qt'.format(k): quantiles[0], 'grad_step': update_step})
            wandb.log({'{}_b_2qt'.format(k): quantiles[1], 'grad_step': update_step})
            wandb.log({'{}_b_3qt'.format(k): quantiles[2], 'grad_step': update_step})
        
      # layer_name = k[k.find('/')+1:k.rfind('/')-4]
      self.prev_neuron_score = neuron_score_dict
    return log_dict

  def _score2mask(self, activation, param, next_param, key):
    del key, param, next_param
    score = self.estimate_neuron_score(activation)
    return score <= self.dead_neurons_threshold

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


class NeuronRecycler(BaseRecycler):
  """Recycle the weights connected to dead neurons.

  In convolutional neural networks, we consider a feature map as neuron.

  Attributes:
    next_layers: dict key a current layer name, value next layer name.
    init_method_outgoing: method to init outgoing weights (random, zero).
    weight_scaling: if true, scale reinit weights.
    incoming_scale: scalar for incoming weights.
    outgoing_scale: scalar for outgoing weights.
  """

  def __init__(
      self,
      all_layers_names,
      track,
      init_method_outgoing='zero',
      weight_scaling=False,
      incoming_scale=1.0,
      outgoing_scale=1.0,
      network='nature',
      prune_dormant_neurons=False,
      neutralize_dormant_neurons=False,
      dead_thres=0.1,
      mass_thres=10,
      weight_revive_eps=0.01,
      K=5,
      M=10,
      **kwargs,
  ):
    super(NeuronRecycler, self).__init__(all_layers_names, track, **kwargs)
    self.init_method_outgoing = init_method_outgoing
    self.weight_scaling = weight_scaling
    self.incoming_scale = incoming_scale
    self.outgoing_scale = outgoing_scale
    self.prune_dormant_neurons = prune_dormant_neurons
    self.neutralize_dormant_neurons = neutralize_dormant_neurons
    self.dead_thres = dead_thres
    self.weight_revive_eps = weight_revive_eps
    self.K = K
    self.track = track
    # prepare a dict that has pointer to next layer give a layer name
    # this is needed because neuron recycle reinitalizes both sides
    # (incoming and outgoing weights) of a neuron and needs a point to the
    # outgoing weights.
    self.next_layers = {}
    for current_layer, next_layer in zip(
        all_layers_names[:-1], all_layers_names[1:]
    ):
      self.next_layers[current_layer] = next_layer

    # we don't recycle the neurons in the output layer.
    self.reset_layers = self.reset_layers[:-1]

    # if network is resnet, recycle only the incoming/outgoing of the first conv
    # layer in each block and final dense layer
    if network == 'resnet':
      self.reset_layers = []
      for layer in self.all_layers_names:
        if 'Conv_1' in layer or 'Conv_3' in layer or 'dense' in layer:
          self.reset_layers.append(layer)

  def is_reset(self, update_step):
    within_reset_interval = (
        update_step >= self.reset_start_step
        and update_step < self.reset_end_step
    )
    return self.is_update_iter(update_step) and within_reset_interval # TODO debugging

  def is_intermediated_required(self, update_step):
    is_logging = self.is_logging_step(update_step)
    is_update_iter = self.is_update_iter(update_step)
    return is_logging or is_update_iter # TODO debugging

  def update_reset_layers(self, reset_start_layer_idx):
    self.reset_layers = self.all_layers_names[reset_start_layer_idx:]
    self.reset_layers = self.reset_layers[:-1]

  def update_weights(self, intermediates, params, key, opt_state):
    if self.prune_dormant_neurons:
      new_param = self.prune_dead_neurons(
          intermediates, params, key, opt_state
      )
    elif self.neutralize_dormant_neurons:
      new_param, opt_state = self.neutralize_massive_and_dead_neurons(
          intermediates, params, key, opt_state
      )
    else:
      new_param, opt_state = self.recycle_dead_neurons(
          intermediates, params, key, opt_state
      )
    return new_param, opt_state

  def recycle_dead_neurons(self, intermedieates, params, key, opt_state):
    """Recycle dead neurons by reinitalizie incoming and outgoing connections.

    Incoming connections are randomly initalized and outgoing connections
    are zero initalized.
    A featuremap is considered dead when its score is below or equal
    dead neuron threshold.
    Args:
      intermedieates: pytree contains the activations over a batch.
      params: current weights of the model.
      key: used to generate random keys.
      opt_state: state of optimizer.

    Returns:
      new model params after recycling dead neurons.
      opt_state: new state for the optimizer

    Raises: raise error if init_method_outgoing is not one of the following
    (random, zero).
    """
    activations_score_dict = flax.traverse_util.flatten_dict(
        flax.core.frozen_dict.freeze(intermedieates), sep='/'
    )
    param_dict = flax.traverse_util.flatten_dict(params, sep='/')
    # create incoming and outgoing masks and reset bias of dead neurons.
    (
        incoming_mask_dict,
        outgoing_mask_dict,
        incoming_random_keys_dict,
        outgoing_random_keys_dict,
        param_dict,
    ) = self.create_masks(param_dict, activations_score_dict, key)

    params = flax.core.freeze(
        flax.traverse_util.unflatten_dict(param_dict, sep='/')
    )
    incoming_random_keys = flax.core.freeze(
        flax.traverse_util.unflatten_dict(incoming_random_keys_dict, sep='/')
    )
    if self.init_method_outgoing == 'random':
      outgoing_random_keys = flax.core.freeze(
          flax.traverse_util.unflatten_dict(outgoing_random_keys_dict, sep='/')
      )
    # reset incoming weights
    incoming_mask = flax.core.freeze(
        flax.traverse_util.unflatten_dict(incoming_mask_dict, sep='/')
    )
    reinit_fn = functools.partial(
        weight_reinit_random,
        weight_scaling=self.weight_scaling,
        scale=self.incoming_scale,
        weights_type='incoming',
    )
    weight_random_reset_fn = jax.jit(functools.partial(jax.tree_util.tree_map, reinit_fn))
    params = weight_random_reset_fn(params, incoming_mask, incoming_random_keys)

    # reset outgoing weights
    outgoing_mask = flax.core.freeze(
        flax.traverse_util.unflatten_dict(outgoing_mask_dict, sep='/')
    )
    # print(outgoing_mask.__class__) # <class 'flax.core.frozen_dict.FrozenDict'>
    # for k in outgoing_mask.keys():
    #   print(11, k)
    #   for k1 in outgoing_mask[k]:
    #     print(22, k1)
    #     for k2 in outgoing_mask[k][k1]:
    #       print(33, k2)
    #       if k2 == 'kernel':
    #         print(55, outgoing_mask[k][k1][k2].shape)
    # import time
    # time.sleep(22)
    # 11 params
    # 22 Conv_0
    # 33 bias # None
    # 33 kernel
    # 55 (8, 8, 4, 32)
    # 22 Conv_1
    # 33 bias
    # 33 kernel
    # 55 (4, 4, 32, 64)
    # 22 Conv_2
    # 33 bias
    # 33 kernel
    # 55 (3, 3, 64, 64)
    # 22 Dense_0
    # 33 bias
    # 33 kernel
    # 55 (7744, 512)
    # 22 final_layer
    # 33 bias
    # 33 kernel
    # 55 (512, 6)
    if self.init_method_outgoing == 'random':
      reinit_fn = functools.partial(
          weight_reinit_random,
          weight_scaling=self.weight_scaling,
          scale=self.outgoing_scale,
          weights_type='outgoing',
      )
      weight_random_reset_fn = jax.jit(
          functools.partial(jax.tree_util.tree_map, reinit_fn)
      )
      params = weight_random_reset_fn(
          params, outgoing_mask, outgoing_random_keys
      )
    elif self.init_method_outgoing == 'zero':
      weight_zero_reset_fn = jax.jit(
          functools.partial(jax.tree_util.tree_map, weight_reinit_zero)
      )
      params = weight_zero_reset_fn(params, outgoing_mask)
    else:
      raise ValueError(f'Invalid init method: {self.init_method_outgoing}')

    # reset mu, nu of adam optimizer for recycled weights.
    reset_momentum_fn = jax.jit(functools.partial(jax.tree_util.tree_map, reset_momentum))
    incoming_mask = flax.core.FrozenDict({k: v for k, v in incoming_mask.items() if 'Encoder' not in k})
    outgoing_mask = flax.core.FrozenDict({k: v for k, v in outgoing_mask.items() if 'Encoder' not in k})
    # print(0, incoming_mask.keys())frozen_dict_keys(['CriticHead', 'LayerNorm_0', 'dense-1_layernorm_tanh'])
    # for k in incoming_mask.keys():
      # print(11, incoming_mask[k].keys())(['critic0', 'critic1'])(['bias', 'kernel'])
    # import time
    # time.sleep(2222)
    # print(incoming_mask['CriticHead']['critic0']['dense0']['kernel'])
    # print(incoming_mask['CriticHead']['critic0']['dense0']['kernel'].shape)
    # print(isinstance(incoming_mask, flax.core.FrozenDict))
    new_mu = reset_momentum_fn(opt_state[0][1], incoming_mask)
    new_mu = reset_momentum_fn(new_mu, outgoing_mask)
    new_nu = reset_momentum_fn(opt_state[0][2], incoming_mask)
    new_nu = reset_momentum_fn(new_nu, outgoing_mask)
    opt_state_list = list(opt_state)
    opt_state_list[0] = optax.ScaleByAdamState(
        opt_state[0].count, mu=new_mu, nu=new_nu
    )
    opt_state = tuple(opt_state_list)
    return params, opt_state
  
  def prune_dead_neurons(self, intermedieates, params, key, opt_state):
    """Recycle dead neurons by reinitalizie incoming and outgoing connections.

    Incoming connections are randomly initalized and outgoing connections
    are zero initalized.
    A featuremap is considered dead when its score is below or equal
    dead neuron threshold.
    Args:
      intermedieates: pytree contains the activations over a batch.
      params: current weights of the model.
      key: used to generate random keys.
      opt_state: state of optimizer.

    Returns:
      new model params after recycling dead neurons.
      opt_state: new state for the optimizer

    Raises: raise error if init_method_outgoing is not one of the following
    (random, zero).
    """
    activations_score_dict = flax.traverse_util.flatten_dict(
        intermedieates, sep='/'
    )
    param_dict = flax.traverse_util.flatten_dict(params, sep='/')

    # create incoming and outgoing masks and reset bias of dead neurons.
    (
        incoming_mask_dict,
        outgoing_mask_dict,
        incoming_random_keys_dict,
        outgoing_random_keys_dict,
        param_dict,
    ) = self.create_masks(param_dict, activations_score_dict, key)
    return params

  def neutralize_massive_and_dead_neurons(self, intermedieates, params, key, opt_state):
    """Recycle dead neurons by reinitalizie incoming and outgoing connections.

    Incoming connections are randomly initalized and outgoing connections
    are zero initalized.
    A featuremap is considered dead when its score is below or equal
    dead neuron threshold.
    Args:
      intermedieates: pytree contains the activations over a batch.
      params: current weights of the model.
      key: used to generate random keys.
      opt_state: state of optimizer.

    Returns:
      new model params after recycling dead neurons.
      opt_state: new state for the optimizer

    Raises: raise error if init_method_outgoing is not one of the following
    (random, zero).
    """
    activations_score_dict = flax.traverse_util.flatten_dict(
        flax.core.frozen_dict.freeze(intermedieates), sep='/'
    )
    param_dict = flax.traverse_util.flatten_dict(params, sep='/')
    # create incoming and outgoing masks and reset bias of dead neurons.
    (
        param_dict, 
        dead_incoming_mask_dict, 
        dead_outgoing_mask_dict
    ) = self.create_dead_mass_masks(param_dict, activations_score_dict, key)

    params = flax.core.freeze(
        flax.traverse_util.unflatten_dict(param_dict, sep='/')
    )
    dead_incoming_mask = flax.core.freeze(
        flax.traverse_util.unflatten_dict(dead_incoming_mask_dict, sep='/')
    )
    dead_outgoing_mask = flax.core.freeze(
        flax.traverse_util.unflatten_dict(dead_outgoing_mask_dict, sep='/')
    )

    # reset mu, nu of adam optimizer for recycled weights.
    reset_momentum_fn = jax.jit(functools.partial(jax.tree_util.tree_map, reset_momentum))
    dead_incoming_mask = flax.core.FrozenDict({k: v for k, v in dead_incoming_mask.items() if 'Encoder' not in k})
    dead_outgoing_mask = flax.core.FrozenDict({k: v for k, v in dead_outgoing_mask.items() if 'Encoder' not in k})
    new_mu = reset_momentum_fn(opt_state[0][1], dead_incoming_mask)
    new_mu = reset_momentum_fn(new_mu, dead_outgoing_mask)
    new_nu = reset_momentum_fn(opt_state[0][2], dead_incoming_mask)
    new_nu = reset_momentum_fn(new_nu, dead_outgoing_mask)
    opt_state_list = list(opt_state)
    opt_state_list[0] = optax.ScaleByAdamState(
        opt_state[0].count, mu=new_mu, nu=new_nu
    )
    opt_state = tuple(opt_state_list)
    return params, opt_state

  def _score2mask(self, activation, param, next_param, key):
    del key, param, next_param
    score = self.estimate_neuron_score(activation)
    return score <= self.dead_thres

  def create_masks(self, param_dict, activations_dict, key):
    """create the masks for recycled weights based on neurons scores.

    Args:
      param_dict: dict of model params.
      activations_dict: dict of the neuron score of each layer.
      key: used seed for random weights.

    Returns:
      incoming_mask_dict
      outgoing_mask_dict
      ingoing_random_keys_dict
      outgoing_random_keys_dict
      param_dict
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
      # print(param_dict.keys())
      # print(self.reset_layers)
      param_key = k + '/kernel' # NOTE needs to be specified for each algo (if using different network architectures)
      param = param_dict[param_key]
      # This won't work for DRQ, since returned keys can be a list.
      # We don't support that at the moment.
      next_key = self.next_layers[k]
      if isinstance(next_key, list):
        next_key = next_key[0]
      next_param = param_dict[next_key + '/kernel']
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
        next_param_key = next_k + '/kernel'
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

        if self.prune_dormant_neurons: # NOTE (ZW) stop the gradients flowing through dormant neurons
          # NOTE (ZW) Log the magnitude of outgoing weights of dormant neurons
          print('Pruning {} outgoing weights at layer {}'.format(outgoing_mask.sum(), k))

      # reset bias
      bias_key = k + '/bias'
      new_bias = jnp.zeros_like(param_dict[bias_key])
      if self.prune_dormant_neurons:
        new_bias -= 99999999
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
    
  def create_dead_mass_masks(self, param_dict, activations_dict, key):
    """create the masks for recycled weights based on neurons scores.

    Args:
      param_dict: dict of model params.
      activations_dict: dict of the neuron score of each layer.
      key: used seed for random weights.

    Returns:
      incoming_mask_dict
      outgoing_mask_dict
      ingoing_random_keys_dict
      outgoing_random_keys_dict
      param_dict
    """
    # dead_incoming_mask_dict = {
    #     k: {} if p.ndim != 1 else None
    #     for k, p in param_dict.items()
    # }
    # dead_outgoing_mask_dict = {
    #     k: {} if p.ndim != 1 else None
    #     for k, p in param_dict.items()
    # }
    dead_incoming_mask_dict = {
        k: jnp.zeros_like(p) if p.ndim != 1 else None
        for k, p in param_dict.items()
    }
    dead_outgoing_mask_dict = {
        k: jnp.zeros_like(p) if p.ndim != 1 else None
        for k, p in param_dict.items()
    }
    # prepare mask of incoming and outgoing recycled connections
    for k in self.reset_layers:      
      param_key = k + '/kernel' # NOTE needs to be specified for each algo (if using different network architectures)
      param = param_dict[param_key] # (51, 256) CriticHead/critic0/dense0
      next_k = self.next_layers[k]
      next_param_key = next_k + '/kernel'
      next_param = param_dict[next_param_key]

      activation = activations_dict[k + '_act/__call__'][0]
      score = self.estimate_neuron_score(activation)
      top_K_values, _, _, least_KM_indices, M = topK_and_leastKM_elements(score, self.K)
      
      dead_neuron_mask = jnp.zeros_like(score)
      dead_neuron_mask = dead_neuron_mask.at[least_KM_indices].set(1)
      dead_neuron_mask = dead_neuron_mask != 0
      dead_incoming_mask, dead_outgoing_mask = self.create_mask_helper(
          dead_neuron_mask, param, next_param
      )
      dead_incoming_mask_dict[param_key] = dead_incoming_mask
      dead_outgoing_mask_dict[next_param_key] = dead_outgoing_mask
      
      for K in range(self.K):
        mass_thres = top_K_values[-(K+1)]
        mass_neuron_mask = score == mass_thres
        mass_incoming_mask, mass_outgoing_mask = self.create_mask_helper(
            mass_neuron_mask, param, next_param
        )
        # reset incoming weights of massive neurons
        weight_shrink_fn = jax.jit(
          functools.partial(weight_shrink, k=mass_thres)
        )
        shrinked_param = weight_shrink_fn(param, mass_incoming_mask)

        # reset incoming weights of dead neurons
        weight_revive_fn = jax.jit(
            functools.partial(weight_revive, K=K, M=M, eps=self.weight_revive_eps, k=mass_thres)
        )
        key, subkey = random.split(key)
        revive_indices = jax.random.choice(subkey, least_KM_indices, shape=(M,), replace=False)
        least_KM_indices = jnp.array([i for i in least_KM_indices if i not in revive_indices])
        dead_neuron_mask = jnp.zeros_like(score)
        # dead_neuron_mask[revive_indices] = 1
        dead_neuron_mask = dead_neuron_mask.at[revive_indices].set(1)
        dead_neuron_mask = dead_neuron_mask != 0
        param, next_param, key = weight_revive_fn(
            param, next_param, dead_neuron_mask, key, mass_incoming_mask, mass_outgoing_mask
        )

        # replace old weights
        param_dict[param_key] = param
        param_dict[next_param_key] = next_param
        param_dict[param_key] = jnp.where(
          mass_incoming_mask, shrinked_param, param_dict[param_key]
        )

        # reset bias
        bias_key = k + '/bias'
        mass_bias = param_dict[bias_key][mass_neuron_mask][0]
        new_bias = mass_bias / mass_thres
        param_dict[bias_key] = jnp.where(
            dead_neuron_mask, new_bias, param_dict[bias_key]
        )
        param_dict[bias_key] = jnp.where(
            mass_neuron_mask, new_bias, param_dict[bias_key]
        ) # True entities in param_dict[bias_key] will be replaced by new_bias    
    
    return (
        param_dict, 
        dead_incoming_mask_dict, 
        dead_outgoing_mask_dict
    )

  def create_mask_helper(self, neuron_mask, current_param, next_param):
    """generate incoming and outgoing weight mask given dead neurons mask.

    Args:
      neuron_mask: mask of size equals the width of a layer.
      current_param: incoming weights of a layer.
      next_param: outgoing weights of a layer.

    Returns:
      incoming_mask
      outgoing_mask
    """

    def mask_creator(expansion_axis, expansion_axes, param, neuron_mask):
      """create a mask of weight matrix given 1D vector of neurons mask.

      Args:
        expansion_axis: List contains 1 axis. The dimension to expand the mask
          for dense layers (weight shape 2D).
        expansion_axes: List conrtains 3 axes. The dimensions to expand the
          score for convolutional layers (weight shape 4D).
        param: weight.
        neuron_mask: 1D mask that represents dead neurons(features).

      Returns:
        mask: mask of weight.
      """
      if param.ndim == 2:
        axes = expansion_axis
        # flatten layer
        # The size of neuron_mask is the same as the width of last conv layer.
        # This conv layer will be flatten and connected to dense layer.
        # we repeat each value of a feature map to cover the spatial dimension.
        if axes[0] == 1 and (param.shape[0] > neuron_mask.shape[0]):
          num_repeatition = int(param.shape[0] / neuron_mask.shape[0])
          neuron_mask = jnp.repeat(neuron_mask, num_repeatition, axis=0)
      elif param.ndim == 4:
        axes = expansion_axes
      mask = jnp.expand_dims(neuron_mask, axis=tuple(axes))
      for i in range(len(axes)):
        mask = jnp.repeat(mask, param.shape[axes[i]], axis=axes[i])
      return mask

    incoming_mask = mask_creator([0], [0, 1, 2], current_param, neuron_mask)
    outgoing_mask = mask_creator([1], [0, 1, 3], next_param, neuron_mask)
    return incoming_mask, outgoing_mask
