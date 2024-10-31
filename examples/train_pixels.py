import os
import random

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
import time
import socket

from jaxrl.agents import DrQLearner, DrQHLGaussianLearner
from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate
from jaxrl.utils import make_env

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'cheetah-run', 'Environment name.')
flags.DEFINE_string('save_dir', './scratch/general/nfs1/$USER/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
# flags.DEFINE_integer('batch_size', 512, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(5e6), 'Number of environment steps.')
flags.DEFINE_integer('start_training', int(1e3),
                     'Number of environment steps to start training.')
flags.DEFINE_integer(
    'action_repeat', None,
    'Action repeat, if None, uses 2 or PlaNet default values.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('track', False, 'Track experiments with Weights and Biases.')
flags.DEFINE_string('wandb_project_name', "dormant-neuron", "The wandb's project name.")
flags.DEFINE_string('wandb_entity', 'zarzard', "the entity (team) of wandb's project")
config_flags.DEFINE_config_file(
    'config',
    'configs/drq_hlg.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

PLANET_ACTION_REPEAT = {
    'cartpole-swingup': 8,
    'reacher-easy': 4,
    'cheetah-run': 4,
    'finger-spin': 2,
    'ball_in_cup-catch': 4,
    'walker-walk': 2
}


from typing import Any, Dict
from ml_collections import ConfigDict

from absl import flags

def merge_configs(flags_obj: Any, config_dict: ConfigDict) -> Dict[str, Any]:
    """
    Merge absl FLAGS and ml_collections.ConfigDict into a single dictionary.
    
    Args:
        flags_obj: The absl FLAGS object
        config_dict: The ml_collections.ConfigDict object
    
    Returns:
        Dict containing all configuration parameters
    """
    # Convert FLAGS to dictionary with actual values
    flags_dict = {}
    # 获取所有 FLAGS 的值
    for flag_name in dir(flags_obj):
        # 跳过内部属性和方法
        if not flag_name.startswith('_'):
            try:
                # 获取实际的值而不是 Flag 对象
                flags_dict[flag_name] = getattr(flags_obj, flag_name)
            except Exception:
                continue
    
    # Convert ConfigDict to regular dict
    if isinstance(config_dict, ConfigDict):
        config_dict = config_dict.to_dict()
    
    # Merge the dictionaries
    # FLAGS values will override ConfigDict values if there are duplicates
    merged_config = {**config_dict, **flags_dict}
    
    return merged_config


def main(_):
    kwargs = dict(FLAGS.config)
    config = merge_configs(FLAGS, FLAGS.config)
    FLAGS.seed = np.random.randint(0, 100000)
    algo = kwargs.pop('algo')
    run_name = f"{FLAGS.seed}"
    if FLAGS.track:
        import wandb

        wandb.init(
            project=FLAGS.wandb_project_name,
            entity=FLAGS.wandb_entity,
            sync_tensorboard=True,
            notes=socket.gethostname(),
            dir=FLAGS.save_dir,
            config=config,
            job_type="training",
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.config.update({"algo": algo})
    summary_writer = SummaryWriter(
        os.path.join(FLAGS.save_dir, 'tb', str(FLAGS.seed)))

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, 'video', 'train')
        video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    if FLAGS.action_repeat is not None:
        action_repeat = FLAGS.action_repeat
    else:
        action_repeat = PLANET_ACTION_REPEAT.get(FLAGS.env_name, 2)

    kwargs = dict(FLAGS.config)
    gray_scale = kwargs.pop('gray_scale')
    image_size = kwargs.pop('image_size')

    def make_pixel_env(seed, video_folder):
        return make_env(FLAGS.env_name,
                        seed,
                        video_folder,
                        action_repeat=action_repeat,
                        image_size=image_size,
                        frame_stack=3,
                        from_pixels=True,
                        gray_scale=gray_scale)

    env = make_pixel_env(FLAGS.seed, video_train_folder)
    eval_env = make_pixel_env(FLAGS.seed + 42, video_eval_folder)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    algo = kwargs.pop('algo')
    replay_buffer_size = kwargs.pop('replay_buffer_size')
    if algo == 'drq':
        agent = DrQLearner(FLAGS.seed,
                        env.observation_space.sample()[np.newaxis],
                        env.action_space.sample()[np.newaxis], **kwargs)
    elif algo == 'drq_hlg':
        agent = DrQHLGaussianLearner(FLAGS.seed,
                                    env.observation_space.sample()[np.newaxis],
                                    env.action_space.sample()[np.newaxis], **kwargs)

    replay_buffer = ReplayBuffer(
        env.observation_space, env.action_space, replay_buffer_size
        or FLAGS.max_steps // action_repeat)

    eval_returns = []
    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(1, FLAGS.max_steps // action_repeat + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, reward, mask, float(done),
                             next_observation)
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            for k, v in info['episode'].items():
                summary_writer.add_scalar(f'training/{k}', v,
                                          info['total']['timesteps'])

        if i >= FLAGS.start_training:
            batch = replay_buffer.sample(int(config['batch_size']))
            update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f'training/{k}', v, i)
                summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v,
                                          info['total']['timesteps'])
            summary_writer.flush()

            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))
            print('env: {}, seed: {}, step: {}, return: {}'.format(FLAGS.env_name, FLAGS.seed, 
                                                                   info['total']['timesteps'], 
                                                                   eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])


if __name__ == '__main__':
    # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    app.run(main)
