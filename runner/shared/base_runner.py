import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from utils.shared_buffer import SharedReplayBuffer


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """

    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']

            # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir / 'models')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        from algorithms.algorithm.r_mappo import RMAPPO as TrainAlgo
        from algorithms.algorithm.rMACPPOPolicy import RMACPPOPolicy as Policy

        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else \
            self.envs.observation_space[0]

        # policy network
        self.policy = Policy(self.all_args,
                             self.envs.observation_space[0],
                             share_observation_space,
                             self.num_agents,
                             self.envs.action_space[0],
                             device=self.device)

        if self.model_dir is not None:
            self.restore()

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)

        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                         self.num_agents,
                                         self.envs.observation_space[0],
                                         share_observation_space,
                                         self.envs.action_space[0])

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self, adjs):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                     adjs,
                                                     np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                     np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
        # fwq
        next_costs = self.trainer.policy.get_cost_values(np.concatenate(self.buffer.share_obs[-1]),
                                                         adjs,
                                                         np.concatenate(self.buffer.rnn_states_cost[-1]),
                                                         np.concatenate(self.buffer.masks[-1]))
        next_costs = np.array(np.split(_t2n(next_costs), self.n_rollout_threads))
        self.buffer.compute_cost_returns(next_costs, self.trainer.value_normalizer)

    def train(self, adjs):
        """Train policies with data in buffer. """
        self.trainer.prep_training()

        action_dim = 4
        num_agents = self.envs.num_agents
        factor = np.ones((self.episode_length, self.n_rollout_threads, num_agents, action_dim), dtype=np.float32)
        self.buffer.update_factor(factor)
        available_actions = None if self.buffer.available_actions is None \
            else self.buffer.available_actions[:-1].reshape(-1, *self.buffer.available_actions.shape[3:])
        old_actions_logprob, _ = self.trainer.policy.actor.evaluate_actions(
            self.buffer.obs[:-1].reshape(-1, *self.buffer.obs.shape[3:]),
            self.buffer.rnn_states[0:1].reshape(-1, *self.buffer.rnn_states.shape[3:]),
            self.buffer.actions.reshape(-1, *self.buffer.actions.shape[3:]),
            self.buffer.masks[:-1].reshape(-1, *self.buffer.masks.shape[3:]),
            available_actions,
            self.buffer.active_masks[:-1].reshape(-1, *self.buffer.active_masks.shape[3:]))

        train_infos = self.trainer.train(self.buffer, adjs)
        # fwq
        cost_train_infos = []  # mieyou yong
        # random update order

        new_actions_logprob, _ = self.trainer.policy.actor.evaluate_actions(
            self.buffer.obs[:-1].reshape(-1, *self.buffer.obs.shape[3:]),
            self.buffer.rnn_states[0:1].reshape(-1, *self.buffer.rnn_states.shape[3:]),
            self.buffer.actions.reshape(-1, *self.buffer.actions.shape[3:]),
            self.buffer.masks[:-1].reshape(-1, *self.buffer.masks.shape[3:]),
            available_actions,
            self.buffer.active_masks[:-1].reshape(-1, *self.buffer.active_masks.shape[3:]))
        factor = factor * _t2n(torch.exp(new_actions_logprob - old_actions_logprob).reshape(-1,
                                                                                            self.n_rollout_threads,
                                                                                            action_dim))

        self.buffer.after_update()
        return train_infos, cost_train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")
        policy_cost_critic = self.trainer.policy.cost_critic
        torch.save(policy_cost_critic.state_dict(), str(self.save_dir) + "/cost_critic.pt")
        # policy_critic = self.trainer.policy.critic
        # torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        if not self.all_args.use_render:
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
            self.policy.critic.load_state_dict(policy_critic_state_dict)
            policy_cost_critic_state_dict = torch.load(str(self.model_dir) + '/cost_critic.pt')
            self.policy.critic.load_state_dict(policy_cost_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
