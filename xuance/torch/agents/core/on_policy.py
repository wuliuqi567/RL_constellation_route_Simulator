import numpy as np
from tqdm import tqdm
from copy import deepcopy
from argparse import Namespace
from xuance.common import Optional, Union, DummyOnPolicyBuffer, DummyOnPolicyBuffer_Atari
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch import Module
from xuance.torch.utils import split_distributions
from xuance.torch.agents.base import Agent
import time

class OnPolicyAgent(Agent):
    """The core class for on-policy algorithm with single agent.

    Args:
        config: the Namespace variable that provides hyper-parameters and other settings.
        envs: the vectorized environments.
    """
    def __init__(self,
                 config: Namespace,
                 envs: Union[DummyVecEnv, SubprocVecEnv]):
        super(OnPolicyAgent, self).__init__(config, envs)
        self.horizon_size = config.horizon_size
        self.n_epochs = config.n_epochs
        self.n_minibatch = config.n_minibatch
        self.gae_lam = config.gae_lambda
        self.auxiliary_info_shape = None
        self.memory: Optional[DummyOnPolicyBuffer] = None

    def _build_memory(self, auxiliary_info_shape=None):
        self.atari = True if self.config.env_name == "Atari" else False
        Buffer = DummyOnPolicyBuffer_Atari if self.atari else DummyOnPolicyBuffer
        self.buffer_size = self.n_envs * self.horizon_size
        self.batch_size = self.buffer_size // self.n_minibatch
        input_buffer = dict(observation_space=self.observation_space,
                            action_space=self.action_space,
                            auxiliary_shape=auxiliary_info_shape,
                            n_envs=self.n_envs,
                            horizon_size=self.horizon_size,
                            use_gae=self.config.use_gae,
                            use_advnorm=self.config.use_advnorm,
                            gamma=self.gamma,
                            gae_lam=self.gae_lam)
        return Buffer(**input_buffer)

    def _build_policy(self) -> Module:
        raise NotImplementedError

    def get_terminated_values(self, observations_next: np.ndarray, rewards: np.ndarray = None) -> np.ndarray:
        """Returns values for terminated states.

        Parameters:
            observations_next (np.ndarray): The terminal observations.
            rewards (np.ndarray): The rewards for terminated states.

        Returns:
            values_next: The values for terminal states.
        """
        policy_out = self.action(self._process_observation(observations_next))
        values_next = policy_out['values']
        return values_next

    def action(self, observations: np.ndarray,
               return_dists: bool = False, return_logpi: bool = False) -> dict:
        """Returns actions and values.

        Parameters:
            observations (np.ndarray): The observation.
            return_dists (bool): Whether to return dists.
            return_logpi (bool): Whether to return log_pi.

        Returns:
            actions: The actions to be executed.
            values: The evaluated values.
            dists: The policy distributions.
            log_pi: Log of stochastic actions.
        """
        _, policy_dists, values = self.policy(observations)
        actions = policy_dists.stochastic_sample()
        log_pi = policy_dists.log_prob(actions).detach().cpu().numpy() if return_logpi else None
        dists = split_distributions(policy_dists) if return_dists else None
        actions = actions.detach().cpu().numpy()
        if values is None:
            values = 0
        else:
            values = values.detach().cpu().numpy()
        return {"actions": actions, "values": values, "dists": dists, "log_pi": log_pi}

    def get_aux_info(self, policy_output: dict = None) -> dict:
        """Returns auxiliary information.

        Parameters:
            policy_output (dict): The output information of the policy.

        Returns:
            aux_info (dict): The auxiliary information.
        """
        return {}

    def train_epochs(self, n_epochs: int = 1) -> dict:
        indexes = np.arange(self.buffer_size)
        train_info = {}
        for _ in range(n_epochs):
            np.random.shuffle(indexes)
            for start in range(0, self.buffer_size, self.batch_size):
                end = start + self.batch_size
                sample_idx = indexes[start:end]
                samples = self.memory.sample(sample_idx)
                train_info = self.learner.update(**samples)
        return train_info
    
    def train_episode(self, n_epochs: int = 1) -> dict:
        indexes = np.arange(self.memory.cur_buf_size)
        train_info = {}
        for _ in range(n_epochs):
            np.random.shuffle(indexes)
            for start in range(0, self.memory.cur_buf_size, self.batch_size):
                end = start + self.batch_size
                sample_idx = indexes[start:end]
                samples = self.memory.episode_sample(sample_idx)
                train_info = self.learner.update(**samples)
        return train_info

    def train(self, train_steps: int) -> None:
        obs = self.envs.buf_obs
        for _ in tqdm(range(train_steps)):
            step_info = {}
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            policy_out = self.action(obs, return_dists=False, return_logpi=False)
            acts, vals = policy_out['actions'], policy_out['values']
            next_obs, rewards, terminals, trunctions, infos = self.envs.step(acts)
            aux_info = self.get_aux_info()
            self.memory.store(obs, acts, self._process_reward(rewards), vals, terminals, aux_info)
            if self.memory.full:
                vals = self.get_terminated_values(next_obs, rewards)
                for i in range(self.n_envs):
                    if terminals[i]:
                        self.memory.finish_path(0.0, i)
                    else:
                        self.memory.finish_path(vals[i], i)
                train_info = self.train_epochs(self.n_epochs)
                self.log_infos(train_info, self.current_step)
                self.memory.clear()

            self.returns = self.gamma * self.returns + rewards
            obs = deepcopy(next_obs)
            for i in range(self.n_envs):
                if terminals[i] or trunctions[i]:
                    self.ret_rms.update(self.returns[i:i + 1])
                    self.returns[i] = 0.0
                    if self.atari and (~trunctions[i]):
                        pass
                    else:
                        if terminals[i]:
                            self.memory.finish_path(0, i)
                        else:
                            vals = self.get_terminated_values(next_obs, rewards)
                            self.memory.finish_path(vals[i], i)
                        obs[i] = infos[i]["reset_obs"]
                        self.envs.buf_obs[i] = obs[i]
                        self.current_episode[i] += 1
                        if self.use_wandb:
                            step_info[f"Episode-Steps/rank_{self.rank}/env-{i}"] = self.current_episode[i]
                            step_info[f"Train-Episode-Rewards/rank_{self.rank}/env-{i}"] = infos[i]["episode_score"]
                        else:
                            step_info[f"Episode-Steps/rank_{self.rank}"] = {f"env-{i}": infos[i]["episode_step"]}
                            step_info[f"Train-Episode-Rewards/rank_{self.rank}"] = {
                                f"env-{i}": infos[i]["episode_score"]}
                        self.log_infos(step_info, self.current_step)
            self.current_step += self.n_envs

    def test(self, env_fn, test_episodes: int) -> list:
        test_envs = env_fn()
        num_envs = test_envs.num_envs
        videos, episode_videos = [[] for _ in range(num_envs)], []
        current_episode, scores, best_score = 0, [], -np.inf
        obs, infos = test_envs.reset()
        if self.config.render_mode == "rgb_array" and self.render:
            images = test_envs.render(self.config.render_mode)
            for idx, img in enumerate(images):
                videos[idx].append(img)
        elif self.config.render_mode == "human" and self.render:
            ret = test_envs.render(self.config.render_mode)
            print("ret", ret)
        all_time_delay_sp = []
        all_time_delay_rl = []
        all_time_bd_sp = []
        all_time_bd_rl = []
        bd_demand = []
        congest_level = []
        all_time_sp_path = []
        all_time_rl_path = []
        consumed_time = []
        start_time = time.time()
        while current_episode < test_episodes:
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            policy_out = self.action(obs)
            next_obs, rewards, terminals, trunctions, infos = test_envs.step(policy_out['actions'])
            if self.config.render_mode == "rgb_array" and self.render:
                images = test_envs.render(self.config.render_mode)
                for idx, img in enumerate(images):
                    videos[idx].append(img)
            # elif self.config.render_mode == "human" and self.render:
            #     ret = test_envs.render(self.config.render_mode)
            #     print("ret", ret)

            obs = deepcopy(next_obs)
            
            for i in range(num_envs):
                if terminals[i] or trunctions[i]:
                    if self.atari and (~trunctions[i]):
                        pass
                    else:
                        obs[i] = infos[i]["reset_obs"]
                        scores.append(infos[i]["episode_score"])
                        current_episode += 1
                        if best_score < infos[i]["episode_score"]:
                            best_score = infos[i]["episode_score"]
                            episode_videos = videos[i].copy()
                        if self.config.test_mode:
                            print("terminal:", terminals[i], "trunction:", trunctions[i])
                            print("Episode: %d, Score: %.2f" % (current_episode, infos[i]["episode_score"]))
                            
                            if "shortest_path" in infos[i]:
                                end_time = time.time()
                                consumed_time.append(end_time - start_time)
                                print(f"Episode {current_episode} consumed time: {end_time - start_time:.2f} seconds")
                                
                                sp_path = infos[i]["shortest_path"] 
                                rl_path = infos[i]["selected_path"] 
                                
                                sp_link_bd = infos[i]["sp_link_bd"]
                                sp_delay = infos[i]["sp_delay"] * 1000
                                
                                rl_link_bd = infos[i]["rl_link_bd"]
                                rl_delay = infos[i]["rl_delay"] * 1000
                                
                                flow_bd_demand = infos[i]["demand"]
                                env_congest_level = infos[i]["congest_level"]
                                
                                all_time_sp_path.append(sp_path)
                                all_time_rl_path.append(rl_path)
                                all_time_delay_sp.append(sp_delay)
                                all_time_delay_rl.append(rl_delay)
                                all_time_bd_sp.append(sp_link_bd)
                                all_time_bd_rl.append(rl_link_bd)
                                
                                congest_level.append(env_congest_level)
                                bd_demand.append(flow_bd_demand)
                                print("sp_path:", sp_path)
                                print("rl_path:", rl_path)
                                print("sp_link_bd:", sp_link_bd)
                                print("rl_link_bd:", rl_link_bd)
                                print("flow_bd_demand:", flow_bd_demand)
                                print("congest_level:", env_congest_level)
                                print("sp_delay:", sp_delay)
                                print("rl_delay:", rl_delay)
                                print()
                                start_time = end_time
        # 画图
        if len(all_time_delay_sp)>0:
            
            print("average consumed time:", np.mean(consumed_time))
            
            import matplotlib.pyplot as plt
            
            # 将所有的数据保存到csv文件中
            import pandas as pd
            data = {
                "Time_Step": range(len(all_time_delay_sp)),
                "Flow_Bandwidth_Demand": bd_demand,
                "Shortest_Path_Delay": all_time_delay_sp,
                "RL_Path_Delay": all_time_delay_rl,
                "Shortest_Path_Bandwidth": all_time_bd_sp,
                "RL_Path_Bandwidth": all_time_bd_rl,
                
            }
            constellation_name = self.config.constellation_name
            algo = self.config.agent
            df = pd.DataFrame(data)
            
            # Create directory if it does not exist
            import os
            if not os.path.exists(f"./{algo}/{constellation_name}"):
                os.makedirs(f"./{algo}/{constellation_name}")
            
            df.to_csv(f"./{algo}/{constellation_name}/data_cl_{congest_level[0]}_bdemand_{bd_demand[0]}.csv", index=False)
            print("Data saved to CSV file. ")
            
            # 将路径信息存入 .txt文件
            with open(f"./{algo}/{constellation_name}/rl_paths_cl_{congest_level[0]}_bdemand_{bd_demand[0]}.txt", "w") as f:
                # f.write("Shortest Path: " + str(sp_path) + "\n")
                for each_path in all_time_rl_path:
                    f.write("RL Path: " + str(each_path) + "\n")
            # 将路径信息存入 .txt文件
            with open(f"./{algo}/{constellation_name}/sp_paths_cl_{congest_level[0]}_bdemand_{bd_demand[0]}.txt", "w") as f:
                # f.write("Shortest Path: " + str(sp_path) + "\n")
                for each_path in all_time_sp_path:
                    f.write("sp Path: " + str(each_path) + "\n")
            
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            x = range(len(all_time_delay_sp))
            
            # Plot delays
            ax1.set_title("Delay Comparison")
            ax1.plot(x, all_time_delay_sp, label='Shortest Path Delay')
            ax1.plot(x, all_time_delay_rl, label='RL Path Delay')
            ax1.set_xlabel('Time Steps')
            ax1.set_ylabel('Delay (s)')
            ax1.legend()
            print("Flow_Bandwidth_Demand:", bd_demand[0])
            print("average delay sp:", np.mean(all_time_delay_sp))
            print("average delay rl:", np.mean(all_time_delay_rl))
            # Plot bandwidths
            ax2.set_title("Bandwidth Comparison")
            ax2.plot(x, all_time_bd_sp, label='Shortest Path Bandwidth')
            ax2.plot(x, all_time_bd_rl, label='RL Path Bandwidth')
            ax2.set_xlabel('Time Steps')
            ax2.set_ylabel('Bandwidth')
            ax2.legend()
            print("average bandwidth sp:", np.mean(all_time_bd_sp))
            print("average bandwidth rl:", np.mean(all_time_bd_rl))
            plt.tight_layout()
            plt.savefig(f"./{algo}/{constellation_name}/data_cl_{congest_level[0]}_bdemand_{bd_demand[0]}.png")
            plt.close()
                    

        if self.config.render_mode == "rgb_array" and self.render:
            # time, height, width, channel -> time, channel, height, width
            videos_info = {"Videos_Test": np.array([episode_videos], dtype=np.uint8).transpose((0, 1, 4, 2, 3))}
            self.log_videos(info=videos_info, fps=self.fps, x_index=self.current_step)

        if self.config.test_mode:
            print("Best Score: %.2f" % (best_score))

        test_info = {
            "Test-Episode-Rewards/Mean-Score": np.mean(scores),
            "Test-Episode-Rewards/Std-Score": np.std(scores)
        }
        self.log_infos(test_info, self.current_step)

        test_envs.close()

        return scores

