import os
import random
import time
import socket

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

from jaxrl.agents import (AWACLearner, DDPGLearner, REDQLearner, SACLearner,
                          SACV1Learner, SACHLGLearner)
from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate
from jaxrl.utils import make_env

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment name.')
flags.DEFINE_string('save_dir', "./scratch/general/nfs1/$USER/", 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
# flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_boolean('reset', False, 'Whether to reset periodically')
flags.DEFINE_integer('reset_interval', 200_000, 'Reset time interval')
flags.DEFINE_integer('updates_per_step', 1, 'Gradient updates per step.')
flags.DEFINE_integer('max_steps', int(1e7), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', False, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('track', False, 'Track experiments with Weights and Biases.')
flags.DEFINE_string('wandb_project_name', "dormant-neuron", "The wandb's project name.")
flags.DEFINE_string('wandb_entity', 'zarzard', "the entity (team) of wandb's project")
flags.DEFINE_integer('index', None, "slurm array index")
config_flags.DEFINE_config_file(
    'config',
    'configs/sac_hlg.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

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
    # settings = []
    # for i in [1., 1.5, 2.]:
    #     for j in [True, False]:
    #         for k in [51, 71, 101, 201]:
    #             settings.append([i, j, k])
    # if FLAGS.index is not None:
    #     setting_for_this_idx = settings[int(FLAGS.index)]
    #     FLAGS.config['sigma'], FLAGS.config['backup_entropy'], FLAGS.config['n_logits'] = setting_for_this_idx
    FLAGS.seed = np.random.randint(0, 100000)
    # envs = ['cheetah-run']
    # if FLAGS.index is not None:
    #     FLAGS.env_name = envs[0]

    kwargs = dict(FLAGS.config)
    config = merge_configs(FLAGS, FLAGS.config)

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
        os.path.join(FLAGS.save_dir, run_name))

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, 'video', 'train')
        video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    env = make_env(FLAGS.env_name, FLAGS.seed, video_train_folder)
    eval_env = make_env(FLAGS.env_name, FLAGS.seed + 42, video_eval_folder)
    if algo == 'sac_hlg':
        if FLAGS.env_name == 'cheetah-run':
            kwargs['max_value'] = 1000.
            print('Adjusting Vmax to 1000 for cheetah-run.')
        else:
            kwargs['max_value'] = 100.
            print('Setting Vmax to 100.')

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)

    replay_buffer_size = kwargs.pop('replay_buffer_size')
    def create_new_agent():
        if algo == 'sac':
            agent = SACLearner(FLAGS.seed,
                            env.observation_space.sample()[np.newaxis],
                            env.action_space.sample()[np.newaxis], **kwargs)
        elif algo == 'redq':
            agent = REDQLearner(FLAGS.seed,
                                env.observation_space.sample()[np.newaxis],
                                env.action_space.sample()[np.newaxis],
                                policy_update_delay=FLAGS.updates_per_step,
                                **kwargs)
        elif algo == 'sac_v1':
            agent = SACV1Learner(FLAGS.seed,
                                env.observation_space.sample()[np.newaxis],
                                env.action_space.sample()[np.newaxis], **kwargs)
        elif algo == 'awac':
            agent = AWACLearner(FLAGS.seed,
                                env.observation_space.sample()[np.newaxis],
                                env.action_space.sample()[np.newaxis], **kwargs)
        elif algo == 'ddpg':
            agent = DDPGLearner(FLAGS.seed,
                                env.observation_space.sample()[np.newaxis],
                                env.action_space.sample()[np.newaxis], **kwargs)
        elif algo == 'sac_hlg':
            agent = SACHLGLearner(FLAGS.seed,
                                env.observation_space.sample()[np.newaxis],
                                env.action_space.sample()[np.newaxis], **kwargs)
        else:
            raise NotImplementedError()
        return agent
    agent = create_new_agent()

    replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                 replay_buffer_size or FLAGS.max_steps)

    eval_returns = []
    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
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

            if 'is_success' in info:
                summary_writer.add_scalar(f'training/success',
                                          info['is_success'],
                                          info['total']['timesteps'])

        if i >= FLAGS.start_training:
            for _ in range(FLAGS.updates_per_step):
                batch = replay_buffer.sample(int(config['batch_size']))
                update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    summary_writer.add_scalar(f'training/{k}', v, i)
                summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(config['discount'], agent, eval_env, FLAGS.eval_episodes)

            for k, v in eval_stats.items():
                summary_writer.add_scalar(f'evaluation/average_{k}s', v,
                                          info['total']['timesteps'])
            summary_writer.flush()

            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))
            print('env: {}, step: {}, seed: {}, alg: {}, eval_return: {}'.format(FLAGS.env_name, 
                                                                                 info['total']['timesteps'], 
                                                                                 FLAGS.seed, config['algo'], eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])
            
        if FLAGS.reset and i % FLAGS.reset_interval == 0:
            # create a completely new agent
            agent = create_new_agent()            


if __name__ == '__main__':
    # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.5'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    app.run(main)
