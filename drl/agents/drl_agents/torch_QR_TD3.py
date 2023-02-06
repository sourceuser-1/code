from random import sample
import drl as drl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import Adam

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
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = F.tanh(self.mu(x))
        return mu

class Critic(nn.Module):
    def __init__(self, hidden_size, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim + 1, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim + 1, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class Torch_QR_TD3(drl.Agent):

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
        self.batch_size = 64#128#64
        self.hidden_size = 128#400#128

        self.expectation_num = 100 # number of samples to take expectation over
        self.kr1 = 0.0 # dropout for ist layer of MLP
        self.lower_quantiles = 0.03 # set alpha for CVaR calculation

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
        self.tau = kwargs['tau']#0.05#0.01#0.05
        self.gamma = kwargs['gamma']#0.99
        # Setup Optimizers
        critic_lr = kwargs['critic_lr']#0.002#0.0001#0.002
        actor_lr = kwargs['actor_lr']#0.001#0.0001#0.001
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
        tau_quart_est_batch = Variable(torch.rand(self.expectation_num, 1)).to(self.device)
        tau_quart_est_batch = tau_quart_est_batch.repeat_interleave(self.batch_size, dim=0)

        tau_quart_target_batch = Variable(torch.rand(self.expectation_num, 1)).to(self.device)
        tau_quart_target_batch = tau_quart_target_batch.repeat_interleave(self.batch_size, dim=0)

        next_action_batch = self.target_actor(next_state_batch.float()).repeat(self.expectation_num, 1)
        next_st_batch = torch.cat((next_state_batch.float().repeat(self.expectation_num, 1), tau_quart_target_batch), 1)
        next_state_action_values_1, next_state_action_values_2 = self.target_critic(next_st_batch, next_action_batch.float())

        # RHS of Bellman Equation
        update_state_action_values = torch.min(next_state_action_values_1, next_state_action_values_2)
        expected_state_action_batch = reward_batch.repeat(self.expectation_num, 1) + (self.gamma * (torch.multiply(done_batch.repeat(self.expectation_num, 1) ,update_state_action_values)))

        # LHS of Bellman Equation
        rep_state = torch.cat((state_batch.float().repeat(self.expectation_num, 1), tau_quart_est_batch), 1)
        #rep_action = (action_batch.float().squeeze(2)).repeat(self.expectation_num, 1)
        rep_action = (action_batch.float()).repeat(self.expectation_num, 1)
        state_action_batch_1, state_action_batch_2 = self.critic_model(rep_state, rep_action)

        # Critic Update
        multiplier1 = torch.abs( (( expected_state_action_batch - state_action_batch_1 ).le(0.)).float() - tau_quart_est_batch  ) # |1{z_tau_est - sample >=0} - tau_est|
        multiplier2 = torch.abs( (( expected_state_action_batch - state_action_batch_2 ).le(0.)).float() - tau_quart_est_batch  ) # |1{z_tau_est - sample >=0} - tau_est|     
        
        self.critic_optimizer.zero_grad()
        value_loss1 = multiplier1 * (F.smooth_l1_loss(expected_state_action_batch, state_action_batch_1 ,reduction = 'none'))
        value_loss2 = multiplier2 * (F.smooth_l1_loss(expected_state_action_batch, state_action_batch_2 ,reduction = 'none'))
        value_loss = value_loss1 + value_loss2

        value_loss = value_loss.reshape(self.batch_size, -1)
        value_loss = value_loss.mean(1)
        value_loss = torch.mean(value_loss)
        value_loss.backward()
        self.critic_optimizer.step()
        self.reward_loss = value_loss.item()

        # Actor Update
        self.actor_optimizer.zero_grad()
        cvar = self.CVaR(state_batch, self.kr1)
        #aleatoric_uncty = self.aleatoric(state_batch, next_state_batch, reward_batch, self.kr1)

        policy_loss = - cvar #+ 0.5*aleatoric_uncty
        policy_loss.backward()
        self.actor_optimizer.step()


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
        #reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = torch.Tensor(self.next_state_buffer[batch_indices]).to(self.device)
        done_batch = torch.Tensor(self.done_buffer[batch_indices]).to(self.device)

        self.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        self.soft_update(self.target_actor, self.actor_model, self.tau)
        self.soft_update(self.target_critic, self.critic_model, self.tau)

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
        self.done_buffer[index] = obs_tuple[4]

        self.buffer_counter += 1

    def CVaR(self, state_batch, kr1):
        # use self.expectation_num number of samples to get CvaR
        num_ex = state_batch.shape[0]
        state_batch = state_batch.to(self.device)

        tau_lower_quants_batch = torch.linspace(0.0,self.lower_quantiles,self.expectation_num).unsqueeze(1)
        tau_lower_quants_batch = tau_lower_quants_batch.repeat_interleave(num_ex,0)
        tau_lower_quants_batch = tau_lower_quants_batch.to(self.device)

        rep_state_batch = state_batch.float().repeat(self.expectation_num, 1) # repeat each state self.expectation_num times
        rep_quant_state_batch = torch.cat((rep_state_batch, tau_lower_quants_batch),1)

        action_batch = self.actor_model(rep_state_batch)
        cvar = self.critic_model.Q1(rep_quant_state_batch, action_batch)
        cvar = cvar.reshape(num_ex,-1)
        cvar = torch.mean(cvar, 1) # get cvar for every state in the batch size:(batch_sz X 1)
        cvar = cvar.mean() # get mean of all the cvars
        return cvar

    def aleatoric(self, state_batch, next_state_batch, reward_batch, kr1):

        state_batch = state_batch.float()
        num_ex = state_batch.shape[0]

        rep_next_state_batch = next_state_batch.float().repeat(self.expectation_num, 1)# repeat each state self.expectation_num times
        
        tau_batch = -torch.tensor(self.lower_quantiles)*torch.rand(self.expectation_num, 1) + torch.tensor(self.lower_quantiles)*torch.ones(self.expectation_num, 1)
        rep_tau_batch = tau_batch.repeat_interleave(num_ex, dim=0)
        rep_tau_batch = rep_tau_batch.to(self.device)

        rep_tau_next_state_batch = torch.cat((rep_next_state_batch, rep_tau_batch),1)

        rep_reward_batch = reward_batch.float().repeat(self.expectation_num, 1)
        rep_reward_batch = rep_reward_batch.to(self.device)
        
        action_batch = self.target_actor(rep_next_state_batch)
        term1 = self.target_critic(rep_tau_next_state_batch, action_batch, self.kr1, mode=False)
        term1 = term1.reshape(num_ex,-1)
        term1 = torch.std(term1, 1)
        
        tau_batch = -torch.tensor(self.lower_quantiles)*torch.rand(self.expectation_num, 1) + torch.tensor(self.lower_quantiles)*torch.ones(self.expectation_num, 1)
        rep_tau_batch = tau_batch.repeat_interleave(num_ex, dim=0)
        rep_tau_batch = rep_tau_batch.to(self.device)
        
        rep_state_batch = state_batch.repeat(self.expectation_num, 1)
        rep_tau_state_batch = torch.cat((rep_state_batch, rep_tau_batch),1)

        action_batch = self.actor_model(rep_state_batch)
        term2 = self.critic_model(rep_tau_state_batch, action_batch, kr1, mode=True)
        term2 = term2.reshape(num_ex, self.expectation_num)#, self.expectation_num)
        term2 = term2.std(axis=1)

        term = term2 #+ term1 
        term = term.mean()
    
        return term

    def load(self, env, agent_id, trial):
        """ Load the ML models """
        results_dir = "saved_models"
        
        actor_path = "{}/{}_{}_actor_{}".format(results_dir, env, agent_id, trial)
        critic_path = "{}/{}_{}_critic_{}".format(results_dir, env, agent_id, trial)
        
        self.actor_model.load_state_dict(torch.load(actor_path))   
        self.critic_model.load_state_dict(torch.load(critic_path))        

    def save(self, env, trial):
        """ Save the ML models """
        results_dir = "saved_models"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        actor_path = "{}/{}_TorchQR_TD3-v0_actor_{}".format(results_dir, env, trial)
        critic_path = "{}/{}_TorchQR_TD3-v0_critic_{}".format(results_dir, env, trial)
        torch.save(self.actor_model.state_dict(), actor_path)
        torch.save(self.critic_model.state_dict(), critic_path)