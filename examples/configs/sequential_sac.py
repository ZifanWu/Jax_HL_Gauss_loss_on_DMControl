import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.algo = 'sequential_sac'

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (256, 256)
    config.batch_size = 256
    config.batch_size_statistics = 256
    config.dead_neurons_threshold = 0.025
    config.dormancy_logging_period = 2000

    config.discount = 0.99

    config.tau = 0.005
    config.target_update_period = 1

    config.init_temperature = 1.0
    config.target_entropy = None
    config.backup_entropy = True

    config.replay_buffer_size = None

    return config
