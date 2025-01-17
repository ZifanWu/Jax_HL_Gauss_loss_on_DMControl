import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.algo = 'drq_hlg'
    config.double_q = True
    
    config.probs_MSE = False
    config.value_MSE = False
    
    config.use_layer_norm_in_critic = True
    config.use_batch_norm = False
    config.use_weight_decay_in_critic = True
    config.WD_rate = 0.0001

    config.n_step_trgt = 1
    config.max_value_schedule = 'linear(100., 100., 500000)'

    config.n_logits = 101
    config.sigma = 0.75
    config.min_value = 0.
    config.max_value = 100.
    config.use_entropy = True

    config.batch_size_statistics = 256
    config.dead_neurons_thresholds = [0.0, 0.025, 0.1]
    config.dormancy_logging_period = 2000

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.actor_hidden_dims = (256, 256)
    config.critic_hidden_dims = (256, 256)
    config.cnn_features = (32, 32, 32, 32)
    config.cnn_strides = (2, 1, 1, 1)
    config.cnn_padding = 'VALID'
    config.latent_dim = 50
    config.batch_size = 512

    config.discount = 0.99

    config.tau = 0.005
    config.target_update_period = 1

    config.init_temperature = 0.1
    config.target_entropy = None

    config.replay_buffer_size = 100000

    config.gray_scale = False
    config.image_size = 84

    return config
