import drl as drl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import Adam
from torch.distributions.normal import Normal
from scipy.stats import norm


import sys
import os

from drl.utils.OUActionNoise import OUActionNoise

class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.mu = nn.Linear(hidden_size, num_outputs)
        #self.mu.weight.data.mul_(0.01)
        #self.mu.bias.data.mul_(0.01)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.ln1(self.linear1(x)))
        x = F.relu(self.ln2(self.linear2(x)))
        mu = F.tanh(self.mu(x))
        return mu

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space

        self.linear1 = nn.Linear(num_inputs, hidden_size) 
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size+num_outputs, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)

        self.Q = nn.Linear(hidden_size, 1)
        self.V = nn.Linear(hidden_size, 1)
        #self.V.weight.data.mul_(0.01)
        #self.V.bias.data.mul_(0.01)

    def forward(self, inputs, actions, kr1, mode):
        #x = torch.cat((inputs, tau_quart), 1)
        x = inputs
        x = F.relu(self.ln1(self.linear1(x)))
        x = torch.cat((x, actions), 1)
        x = F.relu(self.ln2(self.linear2(x)))
        x = F.dropout(x, p=kr1, training=mode)
        x = F.relu(self.ln3(self.linear3(x)))
        Q = self.Q(x)
        R = F.relu(self.V(x))
        return Q, R


class Torch_WCPG(drl.Agent):

    def __init__(self, env, **kwargs):
        """ Define all key variables required for all agent """

        # Get env info
        super().__init__(**kwargs)
        self.env = env

        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.upper_bound = env.action_space.high
        self.lower_bound = env.action_space.low

        #
        self.buffer_counter = 0
        self.buffer_capacity = 50000
        self.batch_size = 128
        self.hidden_size = 128
        self.lower_quantiles = 0.05

        self.kr1 = 0.0 # dropout for ist layer of MLP
        
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))
        self.done_buffer = np.zeros((self.buffer_capacity, 1))

        # Setup models
        self.actor_model = Actor(self.hidden_size, self.num_states, self.num_actions).to(self.device)
        self.target_actor = Actor(self.hidden_size, self.num_states, self.num_actions).to(self.device)
        self.critic_model = Critic(self.hidden_size, self.num_states, self.num_actions).to(self.device)
        self.target_critic = Critic(self.hidden_size, self.num_states, self.num_actions).to(self.device)

        # Set target weights to the active model initially
        self.hard_update(self.target_actor, self.actor_model)  # Make sure target is with the same weight
        self.hard_update(self.target_critic, self.critic_model)

        # Used to update target networks
        self.tau = 0.05
        self.gamma = 0.99
        # Setup Optimizers
        critic_lr = 0.001
        actor_lr = 0.001
        self.critic_optimizer = Adam(self.critic_model.parameters(), lr=critic_lr)
        self.actor_optimizer = Adam(self.actor_model.parameters(), lr=actor_lr)

        # Noise term
        std_dev = 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(self.num_actions),
                                      std_deviation=float(std_dev) * np.ones(self.num_actions))

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch, done_batch
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        next_action_batch = self.target_actor(next_state_batch.float())
        next_state_action_values, next_variance_values = self.target_critic(next_state_batch.float(), next_action_batch.float(), self.kr1, False)

        expected_state_action_batch = reward_batch + (self.gamma * done_batch*next_state_action_values)
        
        self.critic_optimizer.zero_grad()

        state_action_batch, variance_batch = self.critic_model((state_batch.float()), (action_batch.float()), self.kr1, True)

        next_variance =  reward_batch**2 + 2*self.gamma*reward_batch*next_state_action_values \
            + (self.gamma**2)*next_variance_values + (self.gamma**2)*(next_state_action_values**2) \
                - (state_action_batch)**2
        #variance_loss = (next_variance_values**2 + variance_batch**2) -2*(torch.multiply(next_variance_values, variance_batch))  
        #print(variance_loss.mean())  
        value_loss = (F.mse_loss(state_action_batch, expected_state_action_batch)).mean() #+ variance_loss.mean()
        variance_loss = (F.mse_loss(variance_batch, next_variance)).mean()
        value_loss = value_loss + variance_loss
        value_loss = torch.mean(value_loss)
        value_loss = torch.mean(value_loss)
        value_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()

        Q_cvar, V_cvar = self.critic_model((state_batch.float()),self.actor_model((state_batch.float())), self.kr1, False)
        policy_loss = -Q_cvar + norm.pdf(self.lower_quantiles)/norm.cdf(self.lower_quantiles)*(V_cvar)

        policy_loss = torch.mean(policy_loss)

        policy_loss = policy_loss.mean()
        self.reward_loss = policy_loss.item()
        policy_loss.backward()
        self.actor_optimizer.step()

    def CVaR(self, state_batch, kr1):
        state_batch = state_batch.to(self.device)
        Q_cvar, V_cvar = self.critic_model((state_batch.float()),self.actor_model((state_batch.float())), self.kr1, False)
        CvaR = Q_cvar - norm.pdf(self.lower_quantiles)/norm.cdf(self.lower_quantiles)*(V_cvar)
        CvaR = CvaR.mean().mean()
        return CvaR

    def train(self):
        """ Method used to train """
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = torch.Tensor(self.state_buffer[batch_indices]).to(self.device)
        action_batch = torch.Tensor(self.action_buffer[batch_indices]).to(self.device)
        reward_batch = torch.Tensor(self.reward_buffer[batch_indices]).to(self.device)
        done_batch = torch.Tensor(self.done_buffer[batch_indices]).to(self.device)
        #reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = torch.Tensor(self.next_state_buffer[batch_indices]).to(self.device)

        self.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        self.soft_update(self.target_actor, self.actor_model, self.tau)
        self.soft_update(self.target_critic, self.critic_model, self.tau)

    def action1(self, state, kr):
        """ Method used to provide the next action """
        state = state.to(self.device)
        sampled_actions = self.actor_model((Variable(state)))
        noise = self.ou_noise()
        noise = np.expand_dims(noise, 0)
        # Adding noise to action
        sampled_actions = sampled_actions.cpu().detach().numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        return [np.squeeze(legal_action)], noise

    def action(self, state, add_noise):
        """ Method used to provide the next action """
        state = state.to(self.device)
        sampled_actions = self.actor_model((Variable(state)))
        noise = self.ou_noise()
        noise = np.expand_dims(noise, 0)
        # Adding noise to action
        sampled_actions = sampled_actions.cpu().detach().numpy() #+ noise
        noise = 0.1*np.random.randn(sampled_actions.shape[0], sampled_actions.shape[1])
        sampled_actions = sampled_actions + add_noise*noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        return [np.squeeze(legal_action)], noise

    def memory(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    def load(self, env, agent_id, trial):
        """ Load the ML models """
        #results_dir = "drivers/saved_models/" + results_dir
        results_dir = "saved_models/"
        
        actor_path = "{}/{}_{}_actor_{}".format(results_dir, env, agent_id, trial)
        critic_path = "{}/{}_{}_critic_{}".format(results_dir, env, agent_id, trial)

        self.actor_model.load_state_dict(torch.load(actor_path))   
        self.critic_model.load_state_dict(torch.load(critic_path))


    def save(self, env, trial):
        """ Save the ML models """
        #results_dir = "inference/saved_models/" + results_dir
        results_dir = "saved_models/"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        actor_path = "{}/{}_Torch_wcpg-v1_actor_{}".format(results_dir, env, trial)
        critic_path = "{}/{}_Torch_wcpg-v1_critic_{}".format(results_dir, env, trial)

        torch.save(self.actor_model.state_dict(), actor_path)
        torch.save(self.critic_model.state_dict(), critic_path)

    
