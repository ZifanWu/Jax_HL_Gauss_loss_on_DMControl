import os
import random
from pathlib import Path

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
import socket

from jaxrl.agents import DrQLearner, DrQHLGaussianLearner, DrQv2Learner
from jaxrl.datasets import ReplayBuffer, NStepReplayBuffer
from jaxrl.evaluation import evaluate
from jaxrl.utils import make_env

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'cheetah-run', 'Environment name.')
flags.DEFINE_integer('cuda_num', 0, 'cuda number')
flags.DEFINE_string('save_dir', '../../scratch/general/vast/$USER/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_boolean('reset', False, 'Whether to reset the critic head periodically')
flags.DEFINE_boolean('redo_critic', False, 'Whether to redo dormant neurons in the critic head periodically.')
flags.DEFINE_boolean('redo_actor', False, 'Whether to redo dormant neurons in the actor head periodically.')
flags.DEFINE_boolean('ntrlize_d_neurons', False, 'Whether to neutralize dead and massive neurons in the critic head periodically.')
flags.DEFINE_integer('reset_interval', 1000, 'Reset time interval')
flags.DEFINE_integer('reset_start_step', int(1e4), 'Reset time interval')
flags.DEFINE_boolean('reset_mass_opt_state', False, 'Whether to reset the optimizer state of massive neurons')
flags.DEFINE_boolean('scale_mu', False, 'Whether to scale or reset to 0 mu of the optimizer state of massive neurons.')
flags.DEFINE_boolean('scale_nu', False, 'Whether to scale or reset to 0 nu of the optimizer state of massive neurons.')

flags.DEFINE_boolean('use_batched_random_crop', True, 'Whether to use DrQ-v1 img augmentation.')
flags.DEFINE_boolean('msepolicy', False, 'Whether to use MSEPolicy.')
flags.DEFINE_boolean('multivariate_normalpolicy', False, 'Whether to use a multivariate_normal policy.')
flags.DEFINE_boolean('m_05', False, 'Whether to use a multivariate_normal policy.')

flags.DEFINE_integer('update_freq', 1, 'Update the agent every _ env steps.')
flags.DEFINE_integer('updates_per_step', 1, 'Gradient updates per step.')
flags.DEFINE_integer('max_steps', int(1e7), 'Number of environment steps.')
flags.DEFINE_integer('start_training', int(2e3),
                     'Number of environment steps to start training.')
flags.DEFINE_integer(
    'action_repeat', None,
    'Action repeat, if None, uses 2 or PlaNet default values.')
flags.DEFINE_boolean('tqdm', False, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('track', False, 'Track experiments with Weights and Biases.')
flags.DEFINE_string('wandb_project_name', "dormant-neuron", "The wandb's project name.")
flags.DEFINE_string('wandb_entity', 'zarzard', "the entity (team) of wandb's project")
flags.DEFINE_integer('index', None, "slurm array index")
config_flags.DEFINE_config_file(
    'config',
    'configs/drq_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)

PLANET_ACTION_REPEAT = {
    'cartpole-swingup': 8,
    'reacher-easy': 4,
    # 'cheetah-run': 4,
    'finger-spin': 2,
    'ball_in_cup-catch': 4,
    'walker-walk': 2
}

from typing import Any, Dict
from ml_collections import ConfigDict

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
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.cuda_num)
    settings = []
    kwargs = dict(FLAGS.config)
    if FLAGS.index is not None:
        # for i in [
        #           ('hopper-hop', (2048, 2048), 0.01, 400, 1, 1), ('hopper-hop', (2048, 2048), 0.01, 200, 1, 1),
        #           ('hopper-hop', (2048, 2048), 0.0001, 200, 1, 1), ('hopper-hop', (2048, 2048), 0.01, 400, 3, 1),
        #           ('hopper-hop', (2048, 2048), 0.01, 100, 1, 4), ('hopper-hop', (2048, 2048), 0.01, 200, 1, 4),
        #           ('hopper-hop', (2048, 2048), 0.01, 400, 1, 4), ('hopper-hop', (2048, 2048), 0.01, 100, 1, 2)
        #           ]:
        for i in ['acrobot-swingup', 'quadruped-walk', 'quadruped-run', 'finger-turn_hard']:
            for j in [True]:
                for k in [1000]: # TODO test RR scaling
                    for l in [1, 3, 10]:
                        for m in [0.01, 0.003]:
                            for n in [True, False]:
                    # for l in [0.025, 0]:
            # for k in [2, 4, 8]:
                # for k in [1e-4, 1e-3, 1e-2]:
                #     for l in [(512, 512), (1024, 1024), (2096, 2096)]:
                # for k in [3]:
            # for j in [(101, 0.75), (51, 1.5)]:
            #     for k in [1, 3]:
                # for l in [True, False]:
            # for k in ['linear(1.0,0.1,500000)', 'linear(0.2,0.2,500000)']:
            # settings.append(i)
                                settings.append([i,j,j,k,l,m,n])
        setting_for_this_idx = settings[int(FLAGS.index)]
        FLAGS.env_name, FLAGS.redo_critic, FLAGS.ntrlize_d_neurons, \
        FLAGS.reset_interval, FLAGS.config['K'], FLAGS.config['weight_revive_eps'],\
        FLAGS.config['NO_K_mass_thres'] = setting_for_this_idx
        # FLAGS.env_name, FLAGS.config['critic_hidden_dims'],\
        # FLAGS.config['WD_rate'], FLAGS.config['n_logits'], FLAGS.config['n_step_trgt'],\
        # FLAGS.updates_per_step = setting_for_this_idx
        # FLAGS.config['max_value'] = FLAGS.config['n_logits']
        # FLAGS.config['actor_hidden_dims'] = FLAGS.config['critic_hidden_dims']
        
        # if FLAGS.config['critic_hidden_dims'][0] >= 1024:
        #     FLAGS.config['critic_lr'] = 1e-4 * 1024 / FLAGS.config['critic_hidden_dims'][0]
        # if FLAGS.config['actor_hidden_dims'][0] >= 1024:
        #     FLAGS.config['actor_lr'] = 1e-4 * 1024 / FLAGS.config['actor_hidden_dims'][0]
        # FLAGS.env_name, FLAGS.config['max_value'], FLAGS.config['n_step_trgt'] = setting_for_this_idx
        # FLAGS.config['replay_buffer_size'] = setting_for_this_idx
    # FLAGS.config['hidden_dims'][1] = 1024

    if FLAGS.env_name == 'quadruped-run': # NOTE
        FLAGS.config['replay_buffer_size'] = 100000
    if 'humanoid' in FLAGS.env_name:
        FLAGS.config['latent_dim'] = 100
        FLAGS.config['actor_lr'] = 8e-5
        FLAGS.config['critic_lr'] = 8e-5

    config = merge_configs(FLAGS, FLAGS.config)
    FLAGS.seed = np.random.randint(0, 100000)
    algo = kwargs.pop('algo')
    run_name = f"{FLAGS.seed}"
    if FLAGS.track:
        import wandb
        if not Path(FLAGS.save_dir).exists():
            os.makedirs(str(FLAGS.save_dir))
        wandb.init(
            project=FLAGS.wandb_project_name,
            entity=FLAGS.wandb_entity,
            # sync_tensorboard=True,
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
    n_step_trgt = kwargs.pop('n_step_trgt')
    def create_new_agent(env, buffer):
        if algo == 'drq':
            agent = DrQLearner(FLAGS.seed, FLAGS.track, buffer, FLAGS.redo_critic, FLAGS.ntrlize_d_neurons,
                               FLAGS.reset_mass_opt_state, FLAGS.scale_mu, FLAGS.scale_nu,
                               env.observation_space.sample()[np.newaxis], env.action_space.sample()[np.newaxis], 
                               FLAGS.reset_interval, FLAGS.reset_start_step, **kwargs)
        elif algo == 'drq_v2':
            agent = DrQv2Learner(FLAGS.seed, FLAGS.track, buffer, FLAGS.redo_critic, FLAGS.redo_actor,
                                 FLAGS.msepolicy, FLAGS.multivariate_normalpolicy,
                                FLAGS.use_batched_random_crop, FLAGS.m_05, env.observation_space.sample()[np.newaxis],
                                env.action_space.sample()[np.newaxis], FLAGS.reset_interval, **kwargs)
        elif algo == 'drq_hlg':
            agent = DrQHLGaussianLearner(FLAGS.seed, FLAGS.track, buffer,
                                        env.observation_space.sample()[np.newaxis],
                                        env.action_space.sample()[np.newaxis], **kwargs)
        return agent

    if n_step_trgt > 1:
        replay_buffer = NStepReplayBuffer(
            env.observation_space, env.action_space, replay_buffer_size
            or FLAGS.max_steps // action_repeat, kwargs['discount'], n_step_trgt)
    else:
        replay_buffer = ReplayBuffer(
            env.observation_space, env.action_space, replay_buffer_size
            or FLAGS.max_steps // action_repeat)
    agent = create_new_agent(env, replay_buffer)

    eval_returns = []
    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(1, FLAGS.max_steps // action_repeat + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation) if not FLAGS.msepolicy else agent.sample_ddpg_actions(observation)
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
                # summary_writer.add_scalar(f'training/{k}', v,
                #                           info['total']['timesteps'])
                wandb.log({f'training/{k}': v, 'global_step': i})

        if i >= FLAGS.start_training:
            # batch = replay_buffer.sample(int(config['batch_size']))
            # update_info = agent.update(batch)
            if i % FLAGS.update_freq == 0:
                for _ in range(FLAGS.updates_per_step):
                    batch = replay_buffer.sample(int(config['batch_size']))
                    update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    # summary_writer.add_scalar(f'training/{k}', v, i)
                    wandb.log({f'training/{k}': v.tolist(), 'global_step': i})
                # summary_writer.flush()

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(config['discount'], agent, eval_env, FLAGS.eval_episodes, FLAGS.msepolicy)

            for k, v in eval_stats.items():
                # summary_writer.add_scalar(f'evaluation/average_{k}s', v.tolist(),
                #                           info['total']['timesteps'])
                wandb.log({f'evaluation/average_{k}s': v.tolist(), 'global_step':  i})
            # summary_writer.flush()

            eval_returns.append(
                (info['total']['timesteps'], eval_stats['return']))
            print('env: {}, seed: {}, alg: {}, frame: {}, return: {}'.format(FLAGS.env_name, FLAGS.seed, config['algo'],
                                                                   info['total']['timesteps'], 
                                                                   eval_stats['return']))
            np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
                       eval_returns,
                       fmt=['%d', '%.1f'])
            
        if FLAGS.reset and i % FLAGS.reset_interval == 0:
            if algo != 'drq_v2':
                # shared enc params: 388416
                # critic head(s) params: 366232
                # actor head params: 286882
                # so we reset roughtly half of the agent (both layer and param wise)
                
                # save encoder parameters
                old_critic_enc = agent.critic.params['SharedEncoder']
                # target critic has its own copy of encoder
                old_target_critic_enc = agent.target_critic.params['SharedEncoder']
                # save encoder optimizer statistics
                old_critic_enc_opt = agent.critic.opt_state_enc
                
                # create new agent: note that the temperature is new as well
                agent = create_new_agent(env, replay_buffer)
                
                # resetting critic: copy encoder parameters and optimizer statistics
                new_critic_params = agent.critic.params.copy(
                    add_or_replace={'SharedEncoder': old_critic_enc})
                agent.critic = agent.critic.replace(params=new_critic_params, 
                                                    opt_state_enc=old_critic_enc_opt)
                
                # resetting actor: actor in DrQ uses critic's encoder
                # note we could have copied enc optimizer here but actor does not affect enc
                new_actor_params = agent.actor.params.copy(
                    add_or_replace={'SharedEncoder': old_critic_enc})
                agent.actor = agent.actor.replace(params=new_actor_params)
                
                # resetting target critic
                new_target_critic_params = agent.target_critic.params.copy(
                    add_or_replace={'SharedEncoder': old_target_critic_enc})
                agent.target_critic = agent.target_critic.replace(
                    params=new_target_critic_params)
            else:
                old_encoder = agent.encoder
                agent = create_new_agent(env, replay_buffer)
                agent.encoder = old_encoder

if __name__ == '__main__':
    # os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'
    # os.environ['XLA_FLAGS'] = '--xla_gpu_force_compilation_parallelism=1'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] ='false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    app.run(main)
