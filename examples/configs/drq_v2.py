import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    config.algo = 'drq_v2'

    config.stddev_clip = 0.3
    config.stddev_schedule = 'linear(1.0,0.1,500000)' # for hard tasks: 'linear(1.0,0.1,2000000)'; for medium tasks: 'linear(1.0,0.1,500000)'; for easy tasks: 'linear(1.0,0.1,100000)'
    config.n_step_trgt = 3

    config.actor_lr = 1e-4 # NOTE DrQ: 3e-4
    config.critic_lr = 1e-4 # NOTE DrQ: 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (1024, 1024) # NOTE DrQ: 256

    config.cnn_features = (32, 32, 32, 32)
    config.cnn_strides = (2, 1, 1, 1)
    config.cnn_padding = 'VALID'
    config.latent_dim = 50 # NOTE 100 for humanoid tasks
    config.batch_size = 256 # NOTE DrQ: 512

    config.batch_size_statistics = 256
    config.dead_neurons_threshold = 0.1
    config.dormancy_logging_period = 2000

    config.discount = 0.99

    config.tau = 0.01 # NOTE DrQ: 0.005
    config.target_update_period = 1

    config.init_temperature = 0.1
    config.target_entropy = None

    config.replay_buffer_size = 1000000 # NOTE DrQ: 100k

    config.gray_scale = False
    config.image_size = 84

    return config
