import drl as drl

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import Adam
from scipy.misc import derivative
from numpy import random
from scipy.special import softmax
import cupy as cp
from scipy.stats import expon, norm
from torch.distributions import Normal
import math
import copy

import sys
import os

from drl.utils.OUActionNoise import OUActionNoise

def build_net(layer_shape, activation, output_activation):
	'''Build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape, h_acti=nn.ReLU, o_acti=nn.ReLU):
		super(Actor, self).__init__()

		layers = [state_dim] + list(hid_shape)
		self.a_net = build_net(layers, h_acti, o_acti)
		self.mu_layer = nn.Linear(layers[-1], action_dim)
		self.log_std_layer = nn.Linear(layers[-1], action_dim)

		self.LOG_STD_MAX = 2
		self.LOG_STD_MIN = -20


	def forward(self, state, deterministic=False, with_logprob=True):
		'''Network with Enforcing Action Bounds'''
		net_out = self.a_net(state)
		mu = self.mu_layer(net_out)
		log_std = self.log_std_layer(net_out)
		log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)  #总感觉这里clamp不利于学习
		std = torch.exp(log_std)
		dist = Normal(mu, std)

		if deterministic: u = mu
		else: u = dist.rsample() #'''reparameterization trick of Gaussian'''#
		a = torch.tanh(u)

		if with_logprob:
			# get probability density of logp_pi_a from probability density of u, which is given by the original paper.
			# logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)

			# Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
			logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
		else:
			logp_pi_a = None

		return a, logp_pi_a



class Q_Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Q_Critic, self).__init__()
		layers = [state_dim + action_dim + 1] + list(hid_shape) + [1]

		self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
		self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)
		q1 = self.Q_1(sa)
		q2 = self.Q_2(sa)
		return q1, q2

class Torch_QR_SAC_v1(drl.Agent):
	def __init__(self, env, **kwargs):
		self.env = env

		self.num_states = env.observation_space.shape[0]
		self.num_actions = env.action_space.shape[0]
		self.upper_bound = env.action_space.high
		self.lower_bound = env.action_space.low
		self.lower_quantiles = 0.01
		self.expectation_num = 100

		#
		self.buffer_counter = 0
		self.buffer_capacity = 50000
		self.batch_size = 128
		self.hidden_size = 128

		self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

		self.state_buffer = np.zeros((self.buffer_capacity, self.num_states))
		self.action_buffer = np.zeros((self.buffer_capacity, self.num_actions))
		self.reward_buffer = np.zeros((self.buffer_capacity, 1))
		self.next_state_buffer = np.zeros((self.buffer_capacity, self.num_states))
		self.done_buffer = np.zeros((self.buffer_capacity, 1))

		action_dim = self.num_actions
		state_dim = self.num_states

		# Used to update target networks
		self.tau = kwargs['tau']#0.009#0.01#kwargs['tau']#0.001
		self.gamma = 0.99#0.95#kwargs['gamma']# 0.95
		# Setup Optimizers
		critic_lr = kwargs['critic_lr']#1e-4#3e-4#0.0001#kwargs['critic_lr']#0.0001#0.002
		actor_lr = kwargs['actor_lr']#1e-4#0.0001#kwargs['actor_lr']#0.0001#0.001
		hid_shape=(self.hidden_size, self.hidden_size, self.hidden_size)

		self.actor = Actor(state_dim, action_dim, hid_shape).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

		self.q_critic = Q_Critic(state_dim, action_dim, hid_shape).to(self.device)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=critic_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)
		# Freeze target networks with respect to optimizers (only update via polyak averaging)
		for p in self.q_critic_target.parameters():
			p.requires_grad = False

		self.alpha = 0.2
		self.adaptive_alpha = True#False #True
		if self.adaptive_alpha:
			# Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
			self.target_entropy = torch.tensor(-action_dim, dtype=float, requires_grad=True, device=self.device)
			# We learn log_alpha instead of alpha to ensure exp(log_alpha)=alpha>0
			self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.device)
			self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=critic_lr)

	def soft_update(self, target, source, tau):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

	def hard_update(self, target, source):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data)


	def action(self, state, deterministic, with_logprob=False):
		# only used when interact with the env
		with torch.no_grad():
			state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
			deterministic = not deterministic
			a, _ = self.actor(state, deterministic, with_logprob)
		return a.cpu().numpy().flatten(), _

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

	def update(self, s, a, r, s_prime, dead_mask):
		#----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
		with torch.no_grad():
			tau_quart_est_batch = Variable(torch.rand(self.expectation_num, 1)).to(self.device)
			tau_quart_est_batch = tau_quart_est_batch.repeat_interleave(self.batch_size, dim=0)

			tau_quart_target_batch = Variable(torch.rand(self.expectation_num, 1)).to(self.device)
			tau_quart_target_batch = tau_quart_target_batch.repeat_interleave(self.batch_size, dim=0)

			s_prime_rep = s_prime.repeat(self.expectation_num, 1)
			a_prime, log_pi_a_prime = self.actor(s_prime_rep)
			s_prime_tau = torch.cat((s_prime_rep, tau_quart_target_batch), 1)
			target_Q1, target_Q2 = self.q_critic_target(s_prime_tau, a_prime)
			target_Q = torch.min(target_Q1, target_Q2)
			r_rep = r.repeat(self.expectation_num, 1)
			dead_mask_rep = dead_mask.repeat(self.expectation_num, 1)
			target_Q = r_rep + ( dead_mask_rep) * self.gamma * (target_Q - self.alpha * log_pi_a_prime) #Dead or Done is tackled by Randombuffer

		# Get current Q estimates
		s_rep = s.repeat(self.expectation_num, 1)
		s_rep_tau = torch.cat((s_rep, tau_quart_est_batch), 1)
		a_rep = a.repeat(self.expectation_num, 1)
		current_Q1, current_Q2 = self.q_critic(s_rep_tau, a_rep)

		self.q_critic_optimizer.zero_grad()
		multiplier1 = torch.abs( (( target_Q- current_Q1 ).le(0.)).float() - tau_quart_est_batch  ) # |1{z_tau_est - sample >=0} - tau_est|     
		value_loss1 = multiplier1 * (F.smooth_l1_loss(target_Q, current_Q1 ,reduction = 'none'))
		multiplier2 = torch.abs( (( target_Q- current_Q2 ).le(0.)).float() - tau_quart_est_batch  ) # |1{z_tau_est - sample >=0} - tau_est|     
		value_loss2 = multiplier2 * (F.smooth_l1_loss(target_Q, current_Q2 ,reduction = 'none'))
		
		value_loss = value_loss1 + value_loss2
		value_loss = value_loss.reshape(self.batch_size, -1)
		value_loss = value_loss.mean(1)
		value_loss = torch.mean(value_loss)
		#print(value_loss)
		value_loss.backward()
		self.q_critic_optimizer.step()
		self.reward_loss = value_loss.item()

		#----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
		# Freeze Q-networks so you don't waste computational effort
		# computing gradients for them during the policy learning step.
		
		cvar, log_pi_a = self.CVaR1(s, 0.0)

		self.actor_optimizer.zero_grad()
		a_loss = (self.alpha * log_pi_a - cvar).mean()
		a_loss.backward()
		self.actor_optimizer.step()

		#for params in self.q_critic.parameters():
			#params.requires_grad = 	True
		#----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ ------------------------------#
		if self.adaptive_alpha:
			# we optimize log_alpha instead of aplha, which is aimed to force alpha = exp(log_alpha)> 0
			# if we optimize aplpha directly, alpha might be < 0, which will lead to minimun entropy.
			alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
			self.alpha_optim.zero_grad()
			alpha_loss.backward()
			self.alpha_optim.step()
			self.alpha = self.log_alpha.exp()

		#----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
		for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def CVaR1(self, state_batch, kr1):

		#for params in self.q_critic.parameters():
			#params.requires_grad = 	False

		state_batch = state_batch.to(self.device)
		num_ex = state_batch.shape[0]

		tau_lower_quants_batch = torch.linspace(0.0,self.lower_quantiles,self.expectation_num).unsqueeze(1)
		tau_lower_quants_batch = tau_lower_quants_batch.repeat_interleave(num_ex,0)
		tau_lower_quants_batch = tau_lower_quants_batch.to(self.device)

		rep_state_batch = state_batch.float().repeat(self.expectation_num, 1) # repeat each state self.expectation_num times
		rep_quant_state_batch = torch.cat((rep_state_batch, tau_lower_quants_batch),1)

		a, log_pi_a = self.actor(rep_state_batch)
		cvar1, cvar2 = self.q_critic(rep_quant_state_batch, a)
		cvar1 = cvar1.reshape(num_ex, -1).mean(1)
		cvar2 = cvar2.reshape(num_ex, -1).mean(1)
		#cvar = torch.min(cvar1, cvar2)
		cvar = cvar1.mean()

		return cvar, log_pi_a

	def CVaR(self, state_batch, kr1):

		#for params in self.q_critic.parameters():
			#params.requires_grad = 	False

		state_batch = state_batch.to(self.device)
		num_ex = state_batch.shape[0]

		tau_lower_quants_batch = torch.linspace(0.0,self.lower_quantiles,self.expectation_num).unsqueeze(1)
		tau_lower_quants_batch = tau_lower_quants_batch.repeat_interleave(num_ex,0)
		tau_lower_quants_batch = tau_lower_quants_batch.to(self.device)

		rep_state_batch = state_batch.float().repeat(self.expectation_num, 1) # repeat each state self.expectation_num times
		rep_quant_state_batch = torch.cat((rep_state_batch, tau_lower_quants_batch),1)

		a, log_pi_a = self.actor(rep_state_batch)
		cvar1, cvar2 = self.q_critic(rep_quant_state_batch, a)
		cvar1 = cvar1.reshape(num_ex, -1).mean(1)
		cvar2 = cvar2.reshape(num_ex, -1).mean(1)
		cvar = torch.min(cvar1, cvar2)
		cvar = cvar.mean()

		return cvar
		
	def save(self, env, trial):
		""" Save the ML models """
		results_dir = "saved_models"
		if not os.path.exists(results_dir):
			os.makedirs(results_dir)
		actor_path = "{}/{}_Torch_QR_SAC-v1_actor_{}".format(results_dir, env, trial)
		critic_path = "{}/{}_Torch_QR_SAC-v1_critic_{}".format(results_dir, env, trial)
		torch.save(self.actor.state_dict(), actor_path)
		torch.save(self.q_critic.state_dict(), critic_path)
		#torch.save(self.critic_model_2.state_dict(), critic_path_b)


	def load(self, env, agent_id, trial):
		""" Load the ML models """
		results_dir = "saved_models"

		actor_path = "{}/{}_{}_actor_{}".format(results_dir, env, agent_id, trial)
		critic_path = "{}/{}_{}_critic_{}".format(results_dir, env, agent_id, trial)
	
		self.actor.load_state_dict(torch.load(actor_path))   
		self.q_critic.load_state_dict(torch.load(critic_path))       
		