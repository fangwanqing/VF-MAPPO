import time
import numpy as np
import torch
from runner.shared.base_runner import Runner
import pandas as pd
import pickle


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

    def run(self):
        adjs = self.warmup()  # 初始化环境

        start = time.time()
        # episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        episodes = 500
        train_log = []  # 保存reward
        first_time_log = []
        # episodes = 1000    # 回合数
        for episode in range(episodes):
            self.envs.reset()
            print(episode)
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, rnn_states_critic, cost_preds, \
                rnn_states_cost = self.collect(step, adjs)

                # *****

                # Obser reward and next obs
                obs, rewards, costs, dones, infos, all_vehicle_waiting_time, lane_veh, lane_veh_ind = self.envs.step(actions)
                if episode == episodes - 1 and step == self.episode_length - 1:
                    first_time_log.append({"all_vehicle_waiting_time": all_vehicle_waiting_time, "lane_veh": lane_veh, "lane_veh_ind": lane_veh_ind})
                    print({"all_vehicle_waiting_time": all_vehicle_waiting_time, "lane_veh": lane_veh, "lane_veh_ind": lane_veh_ind})

                ###fwq
                # obs = np.squeeze(obs, axis=2)
                # rewards = np.expand_dims(rewards, axis=2)
                # rewards = np.squeeze(rewards, axis=-2)
                data = obs, rewards, costs, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, rnn_states_cost  # fwq 插入了cost
                # print("obs", obs)
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute(adjs)
            train_infos, _ = self.train(adjs)

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                # print("self.buffer.obs", self.buffer.obs)
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length     # self.buffer.rewards ndarry(360, 1, 16, 1)
                train_infos["average_episode_costs"] = np.mean(self.buffer.costs) * self.episode_length
                print("average episode rewards is {}, average episode costs is {}".format(train_infos["average_episode_rewards"], train_infos["average_episode_costs"]))
                self.log_train(train_infos, total_num_steps)
                # self.log_env(env_infos, total_num_steps)
                log = {
                    "episode": episode,
                    "reward": train_infos["average_episode_rewards"],
                    "cost": train_infos["average_episode_costs"]
                }
                train_log.append(log)
                train_log_name = self.envs.path_to_work_directory + "/record.csv"
                dd = pd.DataFrame(train_log)
                dd.to_csv(train_log_name)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

        time_log_name = self.envs.path_to_work_directory + "/time_record.pkl"
        file = open(time_log_name, "wb")
        pickle.dump(first_time_log, file)
        self.envs.bulk_log_multi_process()
        # self.envs.downsample_for_system(path_to_log, dic_traffic_env_conf)

    def warmup(self):
        # reset env
        obs = self.envs.reset()[0]  # shape = （1,16,25）   (5, 2, 14)
        adjs = self.envs.reset()[1]
        # obs = np.squeeze(obs, axis=2)
        # replay buffer
        if self.use_centralized_V:  # Ture
            share_obs = obs.reshape(self.n_rollout_threads, -1)  # shape = (1,400）      (5, 28)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents,
                                                            axis=1)  # shape = （1,16,400）  共用一个v网络，所有的智能体能看到所有智能体的观测    (5, 2, 28)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()  # 将第一步的观测存到经验池里，经验池里有所有步的经验
        self.buffer.obs[0] = obs.copy()  # could not broadcast input array from shape (16,1,25) into shape (1,16,25)
        return adjs

    @torch.no_grad()
    def collect(self, step, adjs):  # 采样动作
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic, cost_pred, rnn_states_cost \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              adjs,
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]),
                                              rnn_states_cost=np.concatenate(self.buffer.rnn_states_cost[step]))  # fwq
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # fwq
        cost_pred = np.array(np.split(_t2n(cost_pred), self.n_rollout_threads))
        rnn_states_cost = np.array(np.split(_t2n(rnn_states_cost), self.n_rollout_threads))


        return values, actions, action_log_probs, rnn_states, rnn_states_critic, rnn_states_critic, cost_pred, rnn_states_cost

    def insert(self, data):
        obs, rewards, costs, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, cost_preds, rnn_states_cost = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                             dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
                                                    dtype=np.float32)
        rnn_states_cost[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_cost.shape[3:]),
                                                  dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards,
                           masks, costs=costs, cost_preds=cost_preds, rnn_states_cost=rnn_states_cost)