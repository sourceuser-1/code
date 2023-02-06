import drl as drl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import Adam
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

        self.mu = nn.Linear(hidden_size, num_outputs)
        #self.mu.weight.data.mul_(0.01)
        #self.mu.bias.data.mul_(0.01)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.ln1(self.linear1(x)))
        x = F.relu(self.ln2(self.linear2(x)))
        mu = F.tanh(self.mu(x))
        return mu

class RewardSurrogate(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(RewardSurrogate, self).__init__()
        self.action_space = action_space
        num_outputs = action_space

        self.linear1 = nn.Linear(num_inputs+1+num_outputs, hidden_size) # +1 to account for the quantile value \beta
        self.ln1 = nn.LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)

        self.V = nn.Linear(hidden_size, 1)
        #self.V.weight.data.mul_(0.01)
        #self.V.bias.data.mul_(0.01)

    def forward(self, inputs, actions):
        x = torch.cat((inputs, actions), 1)
        #x = inputs
        x = F.relu(self.ln1(self.linear1(x)))
        x = F.relu(self.ln2(self.linear2(x)))
        #x = F.dropout(x, p=kr1, training=mode)
        x = F.relu(self.ln3(self.linear3(x)))
        V = self.V(x)
        return V

class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, device):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.device = device

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(self.device) #internal state
        #print(x.shape)
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        #print(output.shape, hn.shape, cn.shape, hn[0,:,:].unsqueeze(0).shape)
        hn = hn[0,:,:].unsqueeze(0)
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out

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
        V = self.V(x)
        return V

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


class Torch_QR_model_based_EVT_v1(drl.Agent):

    def __init__(self, env, **kwargs):
        """ Define all key variables required for all agent """

        # Get env info
        self.env = env

        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.upper_bound = env.action_space.high
        self.lower_bound = env.action_space.low

        #
        self.buffer_counter = 0
        self.buffer_capacity = 50000
        self.batch_size = 64
        self.hidden_size = 128

        self.expectation_num = 100 # number of samples to take expectation over
        self.kr1 = 0.0 # dropout for ist layer of MLP
        self.lower_quantiles = kwargs['lower_quantiles'] #0.05#0.03#0.05 # set alpha for CVaR calculation
        self.thresh_quantile = kwargs['thresh_quantile']#0.95#0.98
        self.reward_surr_weight = 0.0

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

        self.reward_surrogate = LSTM1(num_classes=1, input_size = self.num_states+self.num_actions+1, hidden_size=64, num_layers=2, seq_length=1, device=self.device).to(self.device)
        #self.reward_surrogate = RewardSurrogate(self.hidden_size, self.num_states, self.num_actions).to(self.device)
        self.scale_model = Scale_sigma(self.hidden_size, self.num_states, self.num_actions).to(self.device)

        # Set target weights to the active model initially
        self.hard_update(self.target_actor, self.actor_model)  # Make sure target is with the same weight
        self.hard_update(self.target_critic, self.critic_model)

        # Used to update target networks
        self.tau = kwargs['tau'] #0.01
        self.gamma = kwargs['gamma'] #0.99
        # Setup Optimizers
        critic_lr = kwargs['critic_lr'] #0.001
        actor_lr = kwargs['actor_lr'] #0.001
        reward_surrogate_lr = kwargs['actor_lr'] #0.0009#kwargs['actor_lr'] #0.0009
        self.actor_optimizer = Adam(self.actor_model.parameters(), lr=actor_lr)
        self.reward_surrogate_optimizer = Adam(self.reward_surrogate.parameters(), lr=reward_surrogate_lr)
        self.critic_optimizer = Adam(self.critic_model.parameters(), lr=critic_lr)
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
        # See Pseudo Code. 

        with torch.no_grad():

            lower_tau_quart_est_batch = Variable(torch.rand(self.expectation_num, 1)).to(self.device)*(self.thresh_quantile)
            lower_tau_quart_est_batch = lower_tau_quart_est_batch.repeat_interleave(self.batch_size, dim=0)

            higher_tau_quart_est_batch = (1.0 - self.thresh_quantile)*Variable(torch.rand(self.expectation_num, 1)).to(self.device) \
                 + self.thresh_quantile
            higher_tau_quart_est_batch = higher_tau_quart_est_batch.repeat_interleave(self.batch_size, dim=0)

            tau_quart_est_batch = torch.cat((lower_tau_quart_est_batch, higher_tau_quart_est_batch),0)


            tau_quart_target_batch = Variable(torch.rand(self.expectation_num, 1)).to(self.device)
            tau_quart_target_batch = tau_quart_target_batch.repeat_interleave(self.batch_size, dim=0)

            next_state_rep = next_state_batch.float().repeat(self.expectation_num, 1)

            next_action_batch = self.target_actor(next_state_rep)
            next_st_batch = torch.cat((next_state_rep, tau_quart_target_batch), 1)
            lower_next_state_action_values = self.target_critic(next_st_batch, next_action_batch.float(), self.kr1, mode=False)

            scale_batch = self.scale_model(next_state_rep, next_action_batch.float(), self.kr1, mode=False)
            scale_batch = scale_batch.unsqueeze(1).cpu().numpy()
            expon_samples = (torch.Tensor(expon.rvs(scale = scale_batch)).reshape(-1,1)).to(self.device)

            done_batch =done_batch.repeat(self.expectation_num, 1)

            # RHS of Bellman Equation

            if(self.buffer_counter >= (self.buffer_capacity/2)):
                #  To use model based reward surrogate
                #rep_state = torch.cat((state_batch.float().repeat(self.expectation_num, 1), tau_quart_target_batch), 1)
                rep_state =  torch.cat((state_batch.float().repeat(self.expectation_num, 1), tau_quart_target_batch*(0.05) + 0.95), 1)
                rep_action = (action_batch.float()).repeat(self.expectation_num, 1)
                rep_state = torch.cat((rep_state, rep_action), 1)
                rep_state = rep_state.reshape(rep_state.shape[0], 1, rep_state.shape[1])
                rep_reward_batch = self.reward_surrogate(rep_state)
                #rep_reward_batch = self.reward_surrogate(rep_state, rep_action)
                
                #lower_expected_state_action_batch = rep_reward_batch \
                    #+ (self.gamma * done_batch*lower_next_state_action_values)
                lower_expected_state_action_batch = reward_batch.float().repeat(self.expectation_num, 1) \
                    + (self.gamma * done_batch*lower_next_state_action_values)
                higher_expected_state_action_batch = rep_reward_batch \
                    + (self.gamma * done_batch*(lower_next_state_action_values + expon_samples))
                expected_state_action_batch = torch.cat((lower_expected_state_action_batch, higher_expected_state_action_batch), 0)

            else:
                rb = reward_batch.float().repeat(self.expectation_num, 1)
                lower_expected_state_action_batch = rb \
                    + (self.gamma * done_batch*lower_next_state_action_values)
                #lower_expected_state_action_batch = reward_batch.float().repeat(self.expectation_num, 1) \
                    #+ (self.gamma * done_batch*lower_next_state_action_values)
                higher_expected_state_action_batch = rb \
                    + (self.gamma * done_batch*(lower_next_state_action_values + expon_samples))
                expected_state_action_batch = torch.cat((lower_expected_state_action_batch, higher_expected_state_action_batch), 0)

        # LHS of Bellman Equation
        rep_state = torch.cat((state_batch.float().repeat(2*self.expectation_num, 1), tau_quart_est_batch), 1)
        rep_action = (action_batch.float()).repeat(2*self.expectation_num, 1)
        state_action_batch = self.critic_model(rep_state, rep_action, self.kr1, mode=True)

        # Critic Update
        multiplier = torch.abs( (( expected_state_action_batch - state_action_batch ).le(0.)).float() - tau_quart_est_batch  ) # |1{z_tau_est - sample >=0} - tau_est|     
        
        self.critic_optimizer.zero_grad()
        value_loss = multiplier * (F.smooth_l1_loss(expected_state_action_batch, state_action_batch ,reduction = 'none'))
           
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
            z_vals_threshold = self.critic_model(state_threshold, rep_action_batch.float(), self.kr1, mode=True)

        #tau_lower_quantiles = (1-self.thresh_quantile)*Variable(torch.rand(self.expectation_num, 1)).to(self.device) + self.thresh_quantile
        tau_higher_quantiles = (1-self.thresh_quantile)*Variable(torch.rand(self.expectation_num, 1)).to(self.device) + self.thresh_quantile
        tau_higher_quantiles = tau_higher_quantiles.repeat_interleave(self.batch_size, dim=0)
        rep_state = state_batch.float().repeat(self.expectation_num, 1)
        rep_state_threshold = torch.cat((rep_state, tau_higher_quantiles),1)
        
        with torch.no_grad():
            z_vals = self.critic_model(rep_state_threshold, rep_action_batch, self.kr1, mode=True) - z_vals_threshold

        scale = self.scale_model(rep_state, rep_action_batch, self.kr1, mode=True)
        loss_shape_scale = torch.log(scale) - scale*z_vals

        self.scale_optimizer.zero_grad()
        loss_shape_scale = -torch.mean(loss_shape_scale.reshape(self.batch_size, -1).mean(1))
        self.reward_loss = loss_shape_scale.item()
        loss_shape_scale.backward()
        self.scale_optimizer.step()

        # Reward Surrogate Update
        tau_quart_est_batch = Variable(torch.rand(self.expectation_num, 1)).to(self.device)
        tau_quart_est_batch = tau_quart_est_batch.repeat_interleave(self.batch_size, dim=0)
        rep_state = torch.cat((state_batch.float().repeat(self.expectation_num, 1), tau_quart_est_batch), 1)
        rep_action = (action_batch.float()).repeat(self.expectation_num, 1)
        rep_state = torch.cat((rep_state, rep_action), 1)
        rep_state = rep_state.reshape(rep_state.shape[0], 1, rep_state.shape[1])
        est_reward = self.reward_surrogate(rep_state)
        #est_reward = self.reward_surrogate(rep_state, rep_action)

        actual_reward = reward_batch.float().repeat(self.expectation_num, 1)
        multiplier = torch.abs( (( actual_reward - est_reward ).le(0.)).float() - tau_quart_est_batch  ) # |1{z_tau_est - sample >=0} - tau_est|     
        
        self.reward_surrogate.zero_grad()
        reward_loss = multiplier * (F.smooth_l1_loss(actual_reward, est_reward ,reduction = 'none'))
           
        reward_loss = reward_loss.reshape(self.batch_size, -1)
        reward_loss = reward_loss.mean(1)
        reward_loss = torch.mean(reward_loss)
        self.reward_loss = reward_loss.item()
        reward_loss.backward()
        self.reward_surrogate_optimizer.step()
    
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
        reward_batch = -1.0*torch.Tensor(self.reward_buffer[batch_indices]).to(self.device)
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

    def inst_reward_est(self, state_batch):

        state_batch = state_batch.float()
        num_ex = state_batch.shape[0]
        state_batch = state_batch.to(self.device)

        tau_batch = -torch.tensor(0.9)*torch.rand(self.expectation_num, 1) + torch.tensor(0.9)*torch.ones(self.expectation_num, 1)
        rep_tau_batch = tau_batch.repeat_interleave(num_ex, dim=0)
        rep_tau_batch = rep_tau_batch.to(self.device)
        
        rep_state_batch = state_batch.repeat(self.expectation_num, 1)
        rep_tau_state_batch = torch.cat((rep_state_batch, rep_tau_batch),1)

        action_batch = self.actor_model(rep_state_batch)

        rep_state = torch.cat((rep_tau_state_batch, action_batch), 1)
        rep_state = rep_state.reshape(rep_state.shape[0], 1, rep_state.shape[1])

        inst_reward = self.reward_surrogate(rep_state)

        inst_reward = inst_reward.reshape(num_ex, self.expectation_num)#, self.expectation_num)
        inst_reward = inst_reward.std(axis=1)

        inst_reward = inst_reward.mean()

        return inst_reward        

    def CVaR(self, state_batch, kr1):
        # use self.expectation_num number of samples to get CvaR
        state_batch = state_batch.to(self.device)
        num_ex = state_batch.shape[0]

        #tau_lower_quants_batch = torch.linspace(0.0,self.lower_quantiles,self.expectation_num).unsqueeze(1)
        tau_lower_quants_batch = torch.linspace(1-self.lower_quantiles,1.0, self.expectation_num).unsqueeze(1)
        tau_lower_quants_batch = tau_lower_quants_batch.repeat_interleave(num_ex,0)
        tau_lower_quants_batch = tau_lower_quants_batch.to(self.device)

        rep_state_batch = state_batch.float().repeat(self.expectation_num, 1) # repeat each state self.expectation_num times
        rep_quant_state_batch = torch.cat((rep_state_batch, tau_lower_quants_batch),1)

        action_batch = self.actor_model(rep_state_batch)
        cvar = self.critic_model(rep_quant_state_batch, action_batch, self.kr1, mode=False)
        cvar = cvar.reshape(num_ex,-1)
        cvar = torch.mean(cvar, 1) # get cvar for every state in the batch size:(batch_sz X 1)
        cvar = cvar.mean() # get mean of all the cvars
        return cvar

    def load(self, env, agent_id, trial):
        """ Load the ML models """
        #results_dir = "drivers/saved_models/" + results_dir
        results_dir = "saved_models"
        
        critic_path = "{}/{}_{}_critic_{}".format(results_dir, env, agent_id, trial)
        actor_path = "{}/{}_{}_actor_{}".format(results_dir, env, agent_id, trial)
        reward_surr_path = "{}/{}_{}_reward_{}".format(results_dir, env, agent_id, trial)
        scale_path = "{}/{}_{}_scale_{}".format(results_dir, env, agent_id, trial)

        self.critic_model.load_state_dict(torch.load(critic_path)) 
        self.actor_model.load_state_dict(torch.load(actor_path))   
        self.reward_surrogate.load_state_dict(torch.load(reward_surr_path))  
        self.scale_model.load_state_dict(torch.load(scale_path))   

    def save(self, env, trial):
        """ Save the ML models """
        #results_dir = "inference/saved_models/" + results_dir
        results_dir = "saved_models"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        critic_path = "{}/{}_Torch_QR_model_based_EVT-v1_critic_{}".format(results_dir, env, trial)
        actor_path = "{}/{}_Torch_QR_model_based_EVT-v1_actor_{}".format(results_dir, env, trial)
        reward_surr_path = "{}/{}_Torch_QR_model_based_EVT-v1_reward_{}".format(results_dir, env, trial)
        scale_path = "{}/{}_Torch_QR_model_based_EVT-v1_scale_{}".format(results_dir, env, trial)

        torch.save(self.critic_model.state_dict(), critic_path)
        torch.save(self.actor_model.state_dict(), actor_path)
        torch.save(self.reward_surrogate.state_dict(), reward_surr_path)
        torch.save(self.scale_model.state_dict(), scale_path)