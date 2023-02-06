from random import sample
import drl as drl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import Adam
from scipy.stats import genpareto as GPD
from scipy.stats import expon

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

        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)

        self.mu = nn.Linear(hidden_size, num_outputs)
        #self.mu.weight.data.mul_(0.01)
        #self.mu.bias.data.mul_(0.01)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.ln1(self.linear1(x)))
        x = F.relu(self.ln2(self.linear2(x)))
        x = F.relu(self.ln3(self.linear3(x)))
        mu = F.tanh(self.mu(x))
        return mu

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space

        self.linear1 = nn.Linear(num_inputs+1, hidden_size) # +1 to account for the quantile value \beta
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size+num_outputs, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)


        self.linear4 = nn.Linear(num_inputs+1, hidden_size) # +1 to account for the quantile value \beta
        self.ln4 = nn.LayerNorm(hidden_size)

        self.linear5 = nn.Linear(hidden_size+num_outputs, hidden_size)
        self.ln5 = nn.LayerNorm(hidden_size)

        self.linear6 = nn.Linear(hidden_size, hidden_size)
        self.ln6 = nn.LayerNorm(hidden_size)
       

        self.V1 = nn.Linear(hidden_size, 1)
        self.V1.weight.data.mul_(0.01)
        self.V1.bias.data.mul_(0.01)

        self.V2 = nn.Linear(hidden_size, 1)
        self.V2.weight.data.mul_(0.01)
        self.V2.bias.data.mul_(0.01)

    def forward(self, inputs, actions, kr1, mode):
        #x = torch.cat((inputs, tau_quart), 1)
        x = inputs
        x = F.relu(self.ln1(self.linear1(x)))
        x = torch.cat((x, actions), 1)
        x = F.relu(self.ln2(self.linear2(x)))
        x = F.relu(self.ln3(self.linear3(x)))
        V1 = self.V1(x)

        y = inputs
        y = F.relu(self.ln4(self.linear4(y)))
        y = torch.cat((y, actions), 1)
        y = F.relu(self.ln5(self.linear5(y)))
        y = F.relu(self.ln6(self.linear6(y)))  

        V2 = self.V2(y)

        return V1, V2

class Critic_tail(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic_tail, self).__init__()
        self.action_space = action_space
        num_outputs = action_space

        self.linear1 = nn.Linear(num_inputs+1, hidden_size) # +1 to account for the quantile value \beta
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size+num_outputs, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)

        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.ln4 = nn.LayerNorm(hidden_size)

        self.V = nn.Linear(hidden_size, 1)
        self.V.weight.data.mul_(0.01)
        self.V.bias.data.mul_(0.01)

    def forward(self, inputs, actions, kr1, mode):
        #x = torch.cat((inputs, tau_quart), 1)
        x = inputs
        x = F.relu(self.ln1(self.linear1(x)))
        x = torch.cat((x, actions), 1)
        x = F.relu(self.ln2(self.linear2(x)))
        x = F.dropout(x, p=kr1, training=mode)
        x = F.relu(self.ln3(self.linear3(x)))
        #x = F.relu(self.ln4(self.linear4(x)))
        V = self.V(x)
        return V


class Shape_ksi(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Shape_ksi, self).__init__()
        self.action_space = action_space
        num_outputs = action_space

        self.linear1 = nn.Linear(num_inputs+num_outputs, hidden_size) 
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.V = nn.Linear(hidden_size, 1)

    def forward(self, inputs, actions, kr1, mode):
        x = torch.cat((inputs, actions), 1)
        x = F.relu(self.ln1(self.linear1(x)))
        x = F.relu(self.ln2(self.linear2(x)))
        #V = F.relu(self.V(x))
        V = (F.tanh(self.V(x))+1)*5.0
        return V+1e-1

class Scale_sigma(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Scale_sigma, self).__init__()
        self.action_space = action_space
        num_outputs = action_space

        self.linear1 = nn.Linear(num_inputs+num_outputs, hidden_size) 
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.V = nn.Linear(hidden_size, 1)

    def forward(self, inputs, actions, kr1, mode):
        x = torch.cat((inputs, actions), 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        V = F.relu(self.V(x))  # needs to be positive
        V = (F.tanh(self.V(x))+1)*5.0
        return V+1e-1


class Torch_QR_TD3_EVT_v1(drl.Agent):

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
        self.batch_size = 128#128#64
        self.hidden_size = 128#400#128

        self.expectation_num = 100 # number of samples to take expectation over
        self.kr1 = 0.0 # dropout for ist layer of MLP
        self.lower_quantiles = kwargs['lower_quantiles']# 0.05#0.14#0.14#0.05#0.2#0.09#0.1#0.3 # set alpha for CVaR calculation
        self.thresh_quantile =  kwargs['thresh_quantile']# 0.95#0.9#0.95#0.75#0.75  # the quantile level after which we assume lower confidence

        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

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
        self.critic_tail_model = Critic_tail(self.hidden_size, self.num_states, self.num_actions).to(self.device)
        self.target_critic_tail = Critic_tail(self.hidden_size, self.num_states, self.num_actions).to(self.device)

        self.shape_model = Shape_ksi(self.hidden_size, self.num_states, self.num_actions).to(self.device)
        self.scale_model = Scale_sigma(self.hidden_size, self.num_states, self.num_actions).to(self.device)

        # Set target weights to the active model initially
        self.hard_update(self.target_actor, self.actor_model)  # Make sure target is with the same weight
        self.hard_update(self.target_critic, self.critic_model)
        self.hard_update(self.target_critic_tail, self.critic_tail_model)

        # Used to update target networks
        self.tau = kwargs['tau']#0.05#0.01#0.05
        self.gamma = kwargs['gamma']#0.99
        # Setup Optimizers
        critic_lr = kwargs['critic_lr']#0.002#0.0001#0.002
        actor_lr = kwargs['actor_lr']#0.001#0.0001#0.001
        self.critic_optimizer = Adam(self.critic_model.parameters(), lr=critic_lr)
        self.critic_tail_optimizer = Adam(self.critic_tail_model.parameters(), lr=critic_lr)
        self.actor_optimizer = Adam(self.actor_model.parameters(), lr=actor_lr)
        self.shape_optimizer = Adam(self.shape_model.parameters(), lr=actor_lr)
        self.scale_optimizer = Adam(self.scale_model.parameters(), lr=actor_lr)

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

        with torch.no_grad():

            rand_samp = Variable(torch.rand(self.expectation_num, 1)).to(self.device)*(self.thresh_quantile)
            rand_samp = rand_samp.repeat_interleave(self.batch_size, dim=0)

            lower_tau_quart_est_batch = Variable(torch.rand(self.expectation_num, 1)).to(self.device)*(self.thresh_quantile)
            lower_tau_quart_est_batch = lower_tau_quart_est_batch.repeat_interleave(self.batch_size, dim=0)

            higher_tau_quart_est_batch = (1.0 - self.thresh_quantile)*Variable(torch.rand(self.expectation_num, 1)).to(self.device) \
                + self.thresh_quantile
            higher_tau_quart_est_batch = higher_tau_quart_est_batch.repeat_interleave(self.batch_size, dim=0)

            tau_quart_est_batch = torch.cat((lower_tau_quart_est_batch, higher_tau_quart_est_batch),0)

            tau_threshold = (self.thresh_quantile)*Variable(torch.ones(self.batch_size*self.expectation_num, 1)).to(self.device)
            tau_quart_target_batch = Variable(torch.rand(self.expectation_num, 1)).to(self.device) *(self.thresh_quantile)
            tau_quart_target_batch = tau_quart_target_batch.repeat_interleave(self.batch_size, dim=0)

            next_state_rep = next_state_batch.float().repeat(self.expectation_num, 1)
            next_st_batch_thresh = torch.cat((next_state_rep, tau_threshold), 1)
            next_action_batch = self.target_actor(next_state_rep)

            next_st_batch = torch.cat((next_state_rep, tau_quart_target_batch), 1)
            

            thresh_next_state_action_values1, thresh_next_state_action_values2  = \
                self.target_critic(next_st_batch_thresh, next_action_batch.float(), self.kr1, mode=False)
            lower_next_state_action_values1, lower_next_state_action_values2 = \
                self.target_critic(next_st_batch, next_action_batch.float(), self.kr1, mode=False)

            thresh_next_state_action_values = torch.max(thresh_next_state_action_values1, thresh_next_state_action_values2)
            lower_next_state_action_values = torch.max(lower_next_state_action_values1, lower_next_state_action_values2)

            scale_batch = self.scale_model(next_state_rep, next_action_batch.float(), self.kr1, mode=False)
            scale_batch = scale_batch.unsqueeze(1).cpu().numpy()
            expon_samples = (torch.Tensor(expon.rvs(scale = scale_batch)).reshape(-1,1)).to(self.device)

            scaling_factor = (rand_samp>=self.thresh_quantile).float()

            update_term = (thresh_next_state_action_values +  expon_samples)*scaling_factor + \
                (lower_next_state_action_values)*(1-scaling_factor)

            done_batch =done_batch.repeat(self.expectation_num, 1)

            # RHS of Bellman Equation
            '''lower_next_state_action_values = torch.max(lower_next_state_action_values1, lower_next_state_action_values2)
            lower_expected_state_action_batch = reward_batch.repeat(self.expectation_num, 1) \
                + (self.gamma * done_batch*lower_next_state_action_values)
            higher_expected_state_action_batch = reward_batch.repeat(self.expectation_num, 1) \
                + (self.gamma * done_batch*(lower_next_state_action_values + expon_samples))
            expected_state_action_batch = torch.cat((lower_expected_state_action_batch, higher_expected_state_action_batch), 0)'''
            
            lower_expected_state_action_batch = reward_batch.repeat(self.expectation_num, 1) \
                + (self.gamma * done_batch*(update_term))
            higher_expected_state_action_batch = reward_batch.repeat(self.expectation_num, 1) \
                + (self.gamma * done_batch*(update_term))
            expected_state_action_batch = torch.cat((lower_expected_state_action_batch, higher_expected_state_action_batch), 0)


        # LHS of Bellman Equation
        rep_state = torch.cat((state_batch.float().repeat(2*self.expectation_num, 1), tau_quart_est_batch), 1)
        rep_action = (action_batch.float()).repeat(2*self.expectation_num, 1)
        state_action_batch1, state_action_batch2 = self.critic_model(rep_state, rep_action, self.kr1, mode=True)

        # Critic Update
        multiplier1 = torch.abs( (( expected_state_action_batch - state_action_batch1 ).le(0.)).float() - tau_quart_est_batch  ) # |1{z_tau_est - sample >=0} - tau_est|     
        multiplier2 = torch.abs( (( expected_state_action_batch - state_action_batch2 ).le(0.)).float() - tau_quart_est_batch  ) # |1{z_tau_est - sample >=0} - tau_est|     

        self.critic_optimizer.zero_grad()
        value_loss1 = multiplier1 * (F.smooth_l1_loss(expected_state_action_batch, state_action_batch1 ,reduction = 'none'))
        value_loss2 = multiplier2 * (F.smooth_l1_loss(expected_state_action_batch, state_action_batch2 ,reduction = 'none'))
        value_loss = value_loss1 + value_loss2
           
        value_loss = value_loss.reshape(self.batch_size, -1)
        value_loss = value_loss.mean(1)
        value_loss = torch.mean(value_loss)
        #value_loss.backward(retain_graph=True)
        value_loss.backward()
        self.critic_optimizer.step()

        # MLE Estimation of shape and scale parameters
        
        tau_threshold = (self.thresh_quantile)*Variable(torch.ones(self.batch_size*self.expectation_num, 1)).to(self.device)
        state_threshold = torch.cat((state_batch.float().repeat(self.expectation_num, 1), tau_threshold), 1)
        rep_action_batch = action_batch.float().repeat(self.expectation_num, 1)
        with torch.no_grad():
            z_vals_threshold1, z_vals_threshold2 = self.critic_model(state_threshold, rep_action_batch.float(), self.kr1, mode=True)

        #tau_lower_quantiles = (1-self.thresh_quantile)*Variable(torch.rand(self.expectation_num, 1)).to(self.device) + self.thresh_quantile
        tau_higher_quantiles = (1-self.thresh_quantile)*Variable(torch.rand(self.expectation_num, 1)).to(self.device) + self.thresh_quantile
        tau_higher_quantiles = tau_higher_quantiles.repeat_interleave(self.batch_size, dim=0)
        rep_state = state_batch.float().repeat(self.expectation_num, 1)
        rep_state_threshold = torch.cat((rep_state, tau_higher_quantiles),1)
        
        with torch.no_grad():
            excess1, excess2 =self.critic_model(rep_state_threshold, rep_action_batch, self.kr1, mode=True)
            z_vals = torch.max(excess1 - z_vals_threshold1, excess2 - z_vals_threshold2)

        scale = self.scale_model(rep_state, rep_action_batch, self.kr1, mode=True)
        loss_shape_scale = torch.log(scale) - scale*z_vals

        self.scale_optimizer.zero_grad()
        loss_shape_scale = -torch.mean(loss_shape_scale.reshape(self.batch_size, -1).mean(1))
        self.reward_loss = loss_shape_scale.item()
        loss_shape_scale.backward()
        self.scale_optimizer.step()
    
        # Actor Update

        self.actor_optimizer.zero_grad()
        cvar = self.CVaR(state_batch, self.kr1)
        policy_loss = cvar 
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
        reward_batch = -1*torch.Tensor(self.reward_buffer[batch_indices]).to(self.device)
        #reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = torch.Tensor(self.next_state_buffer[batch_indices]).to(self.device)
        done_batch = torch.Tensor(self.done_buffer[batch_indices]).to(self.device)

        self.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        self.soft_update(self.target_actor, self.actor_model, self.tau)
        self.soft_update(self.target_critic, self.critic_model, self.tau)
        self.soft_update(self.target_critic_tail, self.critic_tail_model, self.tau)

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
        state_batch = state_batch.to(self.device)
        num_ex = state_batch.shape[0]

        tau_higher_quants_batch = torch.linspace(1-self.lower_quantiles, 0.999, self.expectation_num).unsqueeze(1)
        tau_higher_quants_batch = tau_higher_quants_batch.repeat_interleave(num_ex,0)
        tau_higher_quants_batch = tau_higher_quants_batch.to(self.device)

        rep_state = state_batch.float().repeat(self.expectation_num, 1)
        rep_thresh_state_batch = torch.cat((rep_state, tau_higher_quants_batch),1)

        action_batch = self.actor_model(rep_state)
        scale = self.scale_model(rep_state, action_batch, self.kr1, mode=False)

        cvar1, cvar2 = self.critic_model(rep_thresh_state_batch, action_batch, self.kr1, mode=False) 
        #cvar = cvar - 0.5*torch.log(1-tau_higher_quants_batch)/scale
        cvar = torch.max(cvar1, cvar2)
        cvar = cvar.reshape(num_ex,-1)
        cvar = torch.mean(cvar, 1) # get cvar for every state in the batch size:(batch_sz X 1)
        cvar = cvar.mean() # get mean of all the cvars
        return cvar

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
        actor_path = "{}/{}_Torch_QR_TD3_EVT-v1_actor_{}".format(results_dir, env, trial)
        critic_path = "{}/{}_Torch_QR_TD3_EVT-v1_critic_{}".format(results_dir, env, trial)
        scale_path = "{}/{}_Torch_QR_TD3_EVT-v1_scale_{}".format(results_dir, env, trial)
        torch.save(self.actor_model.state_dict(), actor_path)
        torch.save(self.critic_model.state_dict(), critic_path)
        torch.save(self.scale_model.state_dict(), scale_path)
