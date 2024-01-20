import numpy as np

# single env
class DummyVecEnv():
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.num_envs = len(env_fns)
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space
        self.path_to_work_directory = env.path_to_work_directory
        self.actions = None
        self.num_agents = env.num_agent

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)    # 同步动作
        return self.step_wait()

    def step_async(self, actions):    # 同步动作
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, costs, dones, infos, all_vehicle_waiting_time, lane_veh, lane_veh_ind = map(np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i] = self.envs[i].reset()

        self.actions = None
        return obs, rews, costs, dones, infos, all_vehicle_waiting_time, lane_veh, lane_veh_ind

    def reset(self):
        obs = [env.reset()[0] for env in self.envs]
        adjs = [env.reset()[1] for env in self.envs]
        return np.array(obs), np.array(adjs)

    def close(self):
        for env in self.envs:
            env.close()


    def bulk_log_multi_process(self):
        for env in self.envs:
            env.bulk_log_multi_process()