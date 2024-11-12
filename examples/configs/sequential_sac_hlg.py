import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.algo = 'sequential_sac_hlg'
    config.double_q = True

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4
    config.adam_eps = 1e-8

    config.hidden_dims = (256, 256)
    config.batch_size = 256
    config.n_logits = 101
    config.sigma = 0.75 # The "Stop Regressing" paper suggests sigma/bin_width to be 0.75, which means (n_logits,sigma)=(101, .75), (51, 1.5),...
    config.min_value = 0.
    config.max_value = 100.

    config.batch_size_statistics = 256
    config.dead_neurons_threshold = 0.025
    config.dormancy_logging_period = 2000

    config.discount = 0.99

    config.tau = 0.005
    config.target_update_period = 1

    config.use_entropy = True
    config.init_temperature = 1.0
    config.target_entropy = None
    config.soft_critic = True

    config.replay_buffer_size = None

    return config
