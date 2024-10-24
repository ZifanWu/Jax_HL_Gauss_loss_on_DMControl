import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.algo = 'sac_hlg'

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (256, 256)
    config.n_logits = 51
    config.sigma = 1.5 # following the "Stop Regressing" paper, here we set sigma/bin_width to 0.75

    config.discount = 0.99

    config.tau = 0.005
    config.target_update_period = 1

    config.init_temperature = 1.0
    config.target_entropy = None
    config.backup_entropy = True

    config.replay_buffer_size = None

    return config
