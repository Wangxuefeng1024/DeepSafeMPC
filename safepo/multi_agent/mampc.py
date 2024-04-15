# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import scipy.optimize
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os
import sys
import wandb
import time
current_file_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_file_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)
from safepo.common.env import make_ma_mujoco_env
from safepo.common.popart import PopArt
from safepo.common.model import MultiAgentActor as Actor, MultiAgentCritic as Critic
from safepo.common.model import MultiAgentPrediction
from safepo.common.buffer import SeparatedReplayBuffer, ModelPredictiveBuffer
from safepo.common.logger import EpochLogger
from safepo.utils.config import multi_agent_args, set_np_formatting, set_seed, multi_agent_velocity_map, isaac_gym_map, multi_agent_goal_tasks


class StateActionDataset(Dataset):
    def __init__(self, states, actions, next_states):
        self.states = states
        self.actions = actions
        self.next_states = next_states

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        current_state_action = np.concatenate([self.states[idx], self.actions[idx]], axis=0)
        next_state = self.next_states[idx]
        return torch.tensor(current_state_action, dtype=torch.float32), torch.tensor(next_state, dtype=torch.float32)

def check(input):
    output = torch.from_numpy(input) if type(input) == np.ndarray else input
    return output

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

class MAPPO_Policy:

    def __init__(self, config, obs_space, cent_obs_space, act_space):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space
        self.share_obs_space = cent_obs_space

        self.actor = Actor(config, self.obs_space, self.act_space, self.config["device"])
        self.critic = Critic(config, self.share_obs_space, self.config["device"])

        # Loss and optimizer
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.config["actor_lr"], eps=self.config["opti_eps"], weight_decay=self.config["weight_decay"]
            )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.config["critic_lr"], eps=self.config["opti_eps"], weight_decay=self.config["weight_decay"]
            )
        


    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks):
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor

class MAPPO_Trainer():

    def __init__(self, config, policy):
        
        self.config = config
        self.tpdv = dict(dtype=torch.float32, device=self.config["device"])
        self.policy = policy

        self.value_normalizer = PopArt(1, device=self.config["device"])

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.config["clip_param"],
                                                                                    self.config["clip_param"])
        error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
        error_original = self.value_normalizer(return_batch) - values

        value_loss_clipped = huber_loss(error_clipped, self.config["huber_delta"])
        value_loss_original = huber_loss(error_original, self.config["huber_delta"])

        value_loss = torch.max(value_loss_original, value_loss_clipped)

        return value_loss.mean()

    def ppo_update(self, sample):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, _ = sample
        old_action_log_probs_batch, adv_targ, value_preds_batch, return_batch, active_masks_batch = [
            check(x).to(**self.tpdv) for x in [
                old_action_log_probs_batch, adv_targ, value_preds_batch, return_batch, active_masks_batch
            ]
]

        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch,
                                                                              rnn_states_batch,
                                                                              rnn_states_critic_batch,
                                                                              actions_batch,
                                                                              masks_batch,
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.config["clip_param"], 1.0 + self.config["clip_param"]) * adv_targ

        if self.config["use_policy_active_masks"]:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()
        (policy_loss - dist_entropy * self.config["entropy_coef"]).backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.config["max_grad_norm"])
        self.policy.actor_optimizer.step()

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.config["value_loss_coef"]).backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.config["max_grad_norm"])

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, logger):
        advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        advantages_copy = advantages.clone()
        mean_advantages = torch.mean(advantages_copy)
        std_advantages = torch.std(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        for _ in range(self.config["learning_iters"]):
            data_generator = buffer.feed_forward_generator(advantages, self.config["num_mini_batch"])

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample)
            logger.store(
                **{
                    "Loss/Loss_reward_critic": value_loss.item(),
                    "Loss/Loss_actor": policy_loss.item(),
                    "Misc/Reward_critic_norm": critic_grad_norm.item(),
                    "Misc/Entropy": dist_entropy.item(),
                    "Misc/Ratio": imp_weights.detach().mean().item(),
                }
            )

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()


class Runner:

    def __init__(self,
                 vec_env,
                 vec_eval_env,
                 config,
                 model_dir=""
                 ):
        self.envs = vec_env
        self.eval_envs = vec_eval_env
        self.config = config
        self.model_dir = model_dir

        self.num_agents = self.envs.num_agents

        self.obs_size = self.envs.observation_space[0].shape[0]
        self.action_size = self.envs.action_space[0].shape[0]
        self.cen_state_size = self.obs_size*self.num_agents + 4
        self.cen_action_size = self.action_size*self.num_agents
        self.predicit_model = MultiAgentPrediction(self.config, self.cen_state_size, self.cen_action_size, device = self.config['device'])
        self.predict_criterion = nn.MSELoss()
        self.predictor_optimizer = optim.Adam(self.predicit_model.parameters(), lr=0.001)
        self._velocity_threshold = 2.522
        torch.autograd.set_detect_anomaly(True)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        current_file_dir = os.path.dirname(os.path.realpath(__file__))
        parent_dir = os.path.dirname(current_file_dir)
        grandparent_dir = os.path.dirname(parent_dir)
        log_directory = grandparent_dir + '/logs'
        self.logger = EpochLogger(
            log_dir = log_directory,
            seed = str(config["seed"]),
        )
        self.save_dir = str(log_directory+'/models_seed{}'.format(self.config["seed"]))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.logger.save_config(config)
        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id]
            po = MAPPO_Policy(config,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        )
            self.policy.append(po)

        # if self.model_dir != "":
        #     self.restore()

        self.trainer = []
        self.buffer = []

        for agent_id in range(self.num_agents):
            tr = MAPPO_Trainer(config, self.policy[agent_id])
            share_observation_space = self.envs.share_observation_space[agent_id]

            bu = SeparatedReplayBuffer(config,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)
        self.mpc_buffer = ModelPredictiveBuffer(config,
                                self.envs.observation_space[agent_id],
                                share_observation_space,
                                self.envs.action_space[agent_id],
                                self.num_agents)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.config["num_env_steps"]) // self.config["episode_length"] // self.config["n_rollout_threads"]

        train_episode_rewards = torch.zeros(1, self.config["n_rollout_threads"], device=self.config["device"])
        train_episode_costs = torch.zeros(1, self.config["n_rollout_threads"], device=self.config["device"])

        eval_rewards=0.0
        eval_costs=0.0
        for episode in range(episodes):
            wandb_rewards = 0
            wandb_cost = 0

            done_episodes_rewards = []
            done_episodes_costs = []

            for step in range(self.config["episode_length"]):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                obs, share_obs, rewards, costs, dones, infos, _ = self.envs.step(actions)

                dones_env = torch.all(dones, dim=1)

                reward_env = torch.mean(rewards, dim=1).flatten()
                cost_env = torch.mean(costs, dim=1).flatten()

                train_episode_rewards += reward_env
                wandb_rewards += reward_env.sum()
                train_episode_costs += cost_env
                wandb_cost += cost_env.sum()
                for t in range(self.config["n_rollout_threads"]):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[:, t].clone())
                        train_episode_rewards[:, t] = 0
                        done_episodes_costs.append(train_episode_costs[:, t].clone())
                        train_episode_costs[:, t] = 0

                data = obs, share_obs, rewards, dones, infos, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic

                self.insert(data)
            wandb.log({'MARL cost': wandb_cost, 'MARL reward': wandb_rewards})
            self.compute()
            self.train()

            total_num_steps = (episode + 1) * self.config["episode_length"] * self.config["n_rollout_threads"]

            if (episode % self.config["save_interval"] == 0 or episode == episodes - 1):
                self.save()
                
            end = time.time()
            
            if episode % self.config["eval_interval"] == 0 and self.config["use_eval"]:
                eval_rewards, eval_costs = self.eval()

            if len(done_episodes_rewards) != 0:
                aver_episode_costs = torch.stack(done_episodes_costs).mean()
                self.return_aver_cost(aver_episode_costs)
        self.mpc_predict_train()
        # for mpc_episode in range(self.config['mpc_episodes']):
        #     mpc_episode_rewards = 0
        #     mpc_episode_costs = 0
        #     obs, share_obs, _ = self.envs.reset()
        #     initial_pos_vel = torch.zeros(self.config["n_rollout_threads"], 4, device=self.config['device'])
        #     input = torch.cat([obs.view(-1, self.config["n_rollout_threads"], self.obs_size*self.num_agents), initial_pos_vel.view(-1, self.config["n_rollout_threads"], 4)], dim=-1)
        #     for step in range(self.config["episode_length"]):
        #         # MPC actions
        #         action, cost = self.mpc_optimization(input, prediction_horizon=self.config['mpc_horizon'])
        #         action_tensor = torch.tensor(action).view(self.num_agents, self.config["n_rollout_threads"], self.action_size).to(self.config['device'])
        #         action = [action_tensor[i] for i in range(action_tensor.shape[0])]
        #         obs, share_obs, rewards, costs, dones, infos, _ = self.envs.step(action)

        #         dones_env = torch.all(dones, dim=1)

        #         reward_env = torch.mean(rewards, dim=1).flatten()
        #         cost_env = torch.mean(costs, dim=1).flatten()

        #         mpc_episode_rewards += reward_env
        #         mpc_episode_costs += cost_env

        #         values_list = [
        #         [sub_list[0]['x_position'], sub_list[0]['y_position'], sub_list[0]['x_velocity'], sub_list[0]['y_velocity']]
        #             for sub_list in infos if len(sub_list) > 0  # Check if the sub-list has at least one directory
        #         ]

        #         # Convert the list of values to a PyTorch tensor with dtype=float32
        #         additional_information = torch.tensor(values_list, dtype=torch.float32).to(self.config['device'])
        #         input = torch.cat([obs.view(-1, self.config["n_rollout_threads"], self.obs_size*self.num_agents), additional_information.view(-1, self.config["n_rollout_threads"], 4)], dim=-1)
        #     # wandb.log({'MPC cost': torch.sum(mpc_episode_costs), 'MPC reward': torch.sum(mpc_episode_rewards)})

    def return_aver_cost(self, aver_episode_costs):
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].return_aver_insert(aver_episode_costs)

    def warmup(self):
        # reset env
        obs, share_obs, _ = self.envs.reset()

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0].copy_(share_obs[:, agent_id])
            if 'Frank'in self.config['env_name']:
                self.buffer[agent_id].obs[0].copy_(obs[agent_id])
            else:
                self.buffer[agent_id].obs[0].copy_(obs[:, agent_id])
        initial_pos_vel = torch.zeros(self.config["n_rollout_threads"], 4, device=self.config['device'])
        self.mpc_buffer.obs[self.mpc_buffer.index].copy_(obs)
        self.mpc_buffer.share_obs[self.mpc_buffer.index].copy_(share_obs)
        self.mpc_buffer.additional_information[self.mpc_buffer.index].copy_(initial_pos_vel)


    @torch.no_grad()
    def collect(self, step):
        value_collector = []
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        rnn_state_critic_collector = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            value_collector.append(value.detach())
            action_collector.append(action.detach())

            action_log_prob_collector.append(action_log_prob.detach())
            rnn_state_collector.append(rnn_state.detach())
            rnn_state_critic_collector.append(rnn_state_critic.detach())
        if self.config["env_name"] == "Safety9|8HumanoidVelocity-v0":
            zeros = torch.zeros(action_collector[-1].shape[0], 1)
            action_collector[-1]=torch.cat((action_collector[-1], zeros), dim=1)
        values = torch.transpose(torch.stack(value_collector), 1, 0)
        rnn_states = torch.transpose(torch.stack(rnn_state_collector), 1, 0)
        rnn_states_critic = torch.transpose(torch.stack(rnn_state_critic_collector), 1, 0)

        return values, action_collector, action_log_prob_collector, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = torch.all(dones, axis=1)

        rnn_states[dones_env == True] = torch.zeros(
            (dones_env == True).sum(), self.num_agents, self.config["recurrent_N"], self.config["hidden_size"], device=self.config["device"])
        rnn_states_critic[dones_env == True] = torch.zeros(
            (dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:], device=self.config["device"])

        masks = torch.ones(self.config["n_rollout_threads"], self.num_agents, 1, device=self.config["device"])
        masks[dones_env == True] = torch.zeros((dones_env == True).sum(), self.num_agents, 1, device=self.config["device"])

        active_masks = torch.ones(self.config["n_rollout_threads"], self.num_agents, 1, device=self.config["device"])
        active_masks[dones == True] = torch.zeros((dones == True).sum(), 1, device=self.config["device"])
        active_masks[dones_env == True] = torch.ones((dones_env == True).sum(), self.num_agents, 1, device=self.config["device"])

        if self.config["env_name"] == "Safety9|8HumanoidVelocity-v0":
            actions[1]=actions[1][:, :8]
        for agent_id in range(self.num_agents):
            if 'Frank'in self.config['env_name']:
                obs_to_insert = obs[agent_id]
            else:
                obs_to_insert = obs[:, agent_id]
            self.buffer[agent_id].insert(share_obs[:, agent_id], obs_to_insert, rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id], actions[agent_id],
                                         action_log_probs[agent_id],
                                         values[:, agent_id], rewards[:, agent_id], masks[:, agent_id], None,
                                         active_masks[:, agent_id], None)
        if self.config['env_name'] == 'Safety2x3HalfCheetahVelocity-v0':
            values_list = [
            [sub_list[0]['x_position'], 0, sub_list[0]['x_velocity'], 0]
            for sub_list in infos if len(sub_list) > 0  # Check if the sub-list has at least one directory
            ]
        else:
            values_list = [
                [sub_list[0]['x_position'], sub_list[0]['y_position'], sub_list[0]['x_velocity'], sub_list[0]['y_velocity']]
                for sub_list in infos if len(sub_list) > 0  # Check if the sub-list has at least one directory
            ]

        # Convert the list of values to a PyTorch tensor with dtype=float32
        additional_information = torch.tensor(values_list, dtype=torch.float32).to(self.config['device'])

        # additional_information = torch.tensor([infos[0][0]['x_position'], infos[0][0]['y_position'], infos[0][0]['x_velocity'], infos[0][0]['y_velocity']]).to(self.config['device'])

        self.mpc_buffer.insert(share_obs, obs, torch.stack(actions), masks, additional_information)

    def train(self):
        action_dim = 1
        factor = torch.ones(self.config["episode_length"], self.config["n_rollout_threads"], action_dim, device=self.config["device"])

        for agent_id in torch.randperm(self.num_agents):
            action_dim=self.buffer[agent_id].actions.shape[-1]

            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])

            old_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                        self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                        self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                        self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                        available_actions,
                                                        self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            self.trainer[agent_id].train(self.buffer[agent_id], logger=self.logger)

            new_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                        self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                        self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                        self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                        available_actions,
                                                        self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))

            action_prod = torch.prod(torch.exp(new_actions_logprob.detach()-old_actions_logprob.detach()).reshape(self.config["episode_length"],self.config["n_rollout_threads"],action_dim), dim=-1, keepdim=True)
            factor = factor*action_prod.detach()
            self.buffer[agent_id].after_update()

    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")
    
    def predictor_save(self):
        model = self.predicit_model
        torch.save(model.state_dict(), str(self.save_dir) + "/predictor_model.pt")

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    def mpc_predict_train(self):
        print('______start to train predictor______')
        # Obtain the tensors
        states = self.mpc_buffer.obs
        actions = self.mpc_buffer.actions
        infos = self.mpc_buffer.additional_information
        state_size = states.shape[-1]
        action_size = actions.shape[-1]

        curr_infos = infos[0: self.mpc_buffer.size]
        next_infos = infos[1: self.mpc_buffer.size+1]
        curr_states = states[0: self.mpc_buffer.size].view(-1, self.config['n_rollout_threads'], state_size*self.num_agents)
        next_states = states[0: self.mpc_buffer.size].view(-1, self.config['n_rollout_threads'], state_size*self.num_agents)
        curr_inputs = torch.cat([curr_states, curr_infos], dim=-1)
        next_inputs = torch.cat([next_states, next_infos], dim=-1)

        # Define dataset
        dataset = StateActionDataset(curr_inputs.view(-1, state_size*self.num_agents+4), actions.view(-1, action_size*self.num_agents), next_inputs.view(-1, state_size*self.num_agents+4))
        data_loader = DataLoader(dataset, batch_size=self.config['mpc_hidden'], shuffle=True)

        # Training loop
        num_epochs = self.config['mpc_episodes'] # Adjust based on your needs
        for epoch in range(num_epochs):
            train_step = 0
            for inputs, targets in data_loader:
                self.predictor_optimizer.zero_grad()
                outputs = self.predicit_model(inputs)
                loss = self.predict_criterion(outputs, targets)
                loss.backward()
                self.predictor_optimizer.step()
                if train_step % 8 == 0:
                    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
                    wandb.log({'Prediction Error': loss.item()})

        self.predictor_save()

    @torch.no_grad()
    def eval(self, eval_episodes=1):
        eval_episode = 0
        eval_episode_rewards = []
        eval_episode_costs = []
        one_episode_rewards = torch.zeros(1, self.config["n_eval_rollout_threads"], device=self.config["device"])
        one_episode_costs = torch.zeros(1, self.config["n_eval_rollout_threads"], device=self.config["device"])

        eval_obs, _, _ = self.eval_envs.reset()

        eval_rnn_states = torch.zeros(self.config["n_eval_rollout_threads"], self.num_agents, self.config["recurrent_N"], self.config["hidden_size"],
                                   device=self.config["device"])
        eval_masks = torch.ones(self.config["n_eval_rollout_threads"], self.num_agents, 1, device=self.config["device"])

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                if 'Frank'in self.config['env_name']:
                    obs_to_eval = eval_obs[agent_id]
                else:
                    obs_to_eval = eval_obs[:, agent_id]
                eval_actions, temp_rnn_state = \
                    self.trainer[agent_id].policy.act(obs_to_eval,
                                                      eval_rnn_states[:, agent_id],
                                                      eval_masks[:, agent_id],
                                                      deterministic=True)
                eval_rnn_states[:, agent_id] = temp_rnn_state
                eval_actions_collector.append(eval_actions)

            if self.config["env_name"] == "Safety9|8HumanoidVelocity-v0":
                zeros = torch.zeros(eval_actions_collector[-1].shape[0], 1)
                eval_actions_collector[-1]=torch.cat((eval_actions_collector[-1], zeros), dim=1)


            eval_obs, _, eval_rewards, eval_costs, eval_dones, _, _ = self.eval_envs.step(
                eval_actions_collector
            )

            reward_env = torch.mean(eval_rewards, dim=1).flatten()
            cost_env = torch.mean(eval_costs, dim=1).flatten()

            one_episode_rewards += reward_env
            one_episode_costs += cost_env

            eval_dones_env = torch.all(eval_dones, dim=1)

            eval_rnn_states[eval_dones_env == True] = torch.zeros(
                (eval_dones_env == True).sum(), self.num_agents, self.config["recurrent_N"], self.config["hidden_size"], device=self.config["device"])

            eval_masks = torch.ones(self.config["n_eval_rollout_threads"], self.num_agents, 1, device=self.config["device"])
            eval_masks[eval_dones_env == True] = torch.zeros((eval_dones_env == True).sum(), self.num_agents, 1,
                                                          device=self.config["device"])

            for eval_i in range(self.config["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(one_episode_rewards[:, eval_i].mean().item())
                    one_episode_rewards[:, eval_i] = 0
                    eval_episode_costs.append(one_episode_costs[:, eval_i].mean().item())
                    one_episode_costs[:, eval_i] = 0

            if eval_episode >= eval_episodes:
                return np.mean(eval_episode_rewards), np.mean(eval_episode_costs)

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1],
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1])
            next_value = next_value.detach()
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)
    
    
    def cost_function(self, state):
        x_velocity, y_velocity = state[-1], state[-2]
        return 1 if x_velocity**2 + y_velocity**2 >= self._velocity_threshold else 0

    # MPC optimization problem with policy and prediction networks
    def mpc_optimization(self, current_state, prediction_horizon):
        # Define the objective function for optimization

        results = []
        optimized_actions = []
        for i in range(self.config["n_rollout_threads"]):
            initial_actions = []
            states = current_state[:, i].view(-1, self.cen_state_size)
            def objective(actions_flat):
                total_cost = 0
                state = torch.tensor(states, dtype=torch.float32)
                actions = actions_flat.reshape((prediction_horizon, self.cen_action_size))
                for i in range(prediction_horizon):
                    action = torch.tensor(actions[i], dtype=torch.float32)
                    # Use the prediction network to forecast the next state
                    state = self.predicit_model(torch.cat([state.squeeze(), action], dim=-1))
                    total_cost += self.cost_function(state.detach().numpy())
                return total_cost
            # Use the policy network to generate the initial action for the optimization
            # initial_actions = []
            # current_state = current_state.view(-1, self.cen_state_size)
            for i in range(self.num_agents):
                obs = states[:, i*self.obs_size:(i+1)*self.obs_size]
                action, action_prob, _ = self.trainer[i].policy.actor(obs, torch.zeros(self.config['buffer_batch']), masks=torch.zeros(self.config['buffer_batch']), deterministic=False )
                initial_actions.append(action.detach().cpu().numpy())
        
            initial_actions = np.array(initial_actions).reshape(-1 ,self.cen_action_size)
            initial_guess = np.tile(initial_actions, prediction_horizon)
            action_bounds = [(-1, 1)] * (prediction_horizon * self.cen_action_size)
            # Solve the optimization problem
            result = scipy.optimize.minimize(objective, initial_guess, method='SLSQP', bounds=action_bounds)
            results.append(result)
            optimized_actions.append(result.x.reshape((prediction_horizon, -1))[0])
        return optimized_actions, results

def train(args, cfg_train):

    if args.task in multi_agent_velocity_map:
        env = make_ma_mujoco_env(
        scenario=args.scenario,
        agent_conf=args.agent_conf,
        seed=args.seed,
        cfg_train=cfg_train,
    )
        cfg_eval = copy.deepcopy(cfg_train)
        cfg_eval["seed"] = args.seed + 10000
        cfg_eval["n_rollout_threads"] = cfg_eval["n_eval_rollout_threads"]
        eval_env = make_ma_mujoco_env(
        scenario=args.scenario,
        agent_conf=args.agent_conf,
        seed=cfg_eval['seed'],
        cfg_train=cfg_eval,
    )
    else: 
        raise NotImplementedError
    
    torch.set_num_threads(4)
    runner = Runner(env, eval_env, cfg_train, args.model_dir)

    if args.model_dir != "":
        runner.eval(100000)
    else:
        runner.run()

if __name__ == '__main__':
    set_np_formatting()
    args, cfg_env, cfg_train = multi_agent_args(algo="mampc")
    # cfg_train['n_rollout_threads'] = 1
    # cfg_train['n_eval_rollout_threads'] = 1
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    wandb.init(config=args,
            project='mampc',
            name='mampc_'+args.task+'_'+str(args.seed),
            job_type = "training",
            reinit = True)
    if args.write_terminal:
        train(args=args, cfg_train=cfg_train)
    else:
        terminal_log_name = "terminal.log"
        error_log_name = "error.log"
        terminal_log_name = f"seed{args.seed}_{terminal_log_name}"
        error_log_name = f"seed{args.seed}_{error_log_name}"
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if not os.path.exists(cfg_train['log_dir']):
            os.makedirs(cfg_train['log_dir'], exist_ok=True)
        with open(
            os.path.join(
                f"{cfg_train['log_dir']}",
                terminal_log_name,
            ),
            "w",
            encoding="utf-8",
        ) as f_out:
            sys.stdout = f_out
            with open(
                os.path.join(
                    f"{cfg_train['log_dir']}",
                    error_log_name,
                ),
                "w",
                encoding="utf-8",
            ) as f_error:
                sys.stderr = f_error
                train(args=args, cfg_train=cfg_train)
