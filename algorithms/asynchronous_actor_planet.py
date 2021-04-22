import torch.nn as nn
from parameter import args
from torch import jit
import torch
from parameter import args
import torch
import os
from torch import nn, optim
from torch.nn import functional as F
from parameter import args
from models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel, ValueModel, ActorModel, MergeModel
from utils import lineplot, write_video, imagine_ahead, lambda_return, FreezeParameters, Save_Txt, ActivateParameters, get_modules
from torch.distributions import Normal
from asynchronous_init_sample import Worker_init_Sample
from torch.multiprocessing import Pipe, Manager
from asynchronous_actor import Worker_actor
from typing import Optional, List
import torch
import time
from torch import jit, nn
from torch.nn import functional as F
import torch.distributions
from torch.distributions.normal import Normal
from torch.distributions import constraints
from torch.distributions.transforms import Transform, TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution
import numpy as np
from parameter import args


# Model-predictive control planner with cross-entropy method and learned transition model
class MPCPlanner(nn.Module):
	__constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates']

	def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates):
		super().__init__()
		# self.transition_model, self.reward_model = transition_model, reward_model
		self.action_size = action_size
		self.top_planning_horizon = planning_horizon
		self.optimisation_iters = optimisation_iters
		self.candidates, self.top_candidates = candidates, top_candidates

	def upper_transition_model(self, prev_state, actions, prev_belief, obs=None, nonterminals=None):
		actions = torch.transpose(actions, 0, 1) if args.MultiGPU and torch.cuda.device_count() > 1 else actions
		nonterminals = torch.transpose(nonterminals, 0, 1).cuda() if args.MultiGPU and torch.cuda.device_count() > 1 and nonterminals is not None else nonterminals
		obs = torch.transpose(obs, 0, 1).cuda() if args.MultiGPU and torch.cuda.device_count() > 1 and obs is not None else obs
		temp_val = self.transition_model(prev_state.cuda(), actions.cuda(), prev_belief.cuda(), obs, nonterminals)
		return list(map(lambda x: x.view(-1, prev_state.shape[0], x.shape[2]), [x for x in temp_val]))

	# @jit.script_method
	def forward(self, belief, state):
		B, H, Z = belief.size(0), belief.size(1), state.size(1)
		belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
		# Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
		action_mean, action_std_dev = torch.zeros(self.top_planning_horizon, B, 1, self.action_size, device=belief.device), torch.ones(self.top_planning_horizon, B, 1, self.action_size, device=belief.device)
		for _ in range(self.optimisation_iters):
			# print("optimization_iters",_)
			# Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
			actions = (action_mean + action_std_dev * torch.randn(self.top_planning_horizon, B, self.candidates, self.action_size, device=action_mean.device)).view(self.top_planning_horizon, B * self.candidates, self.action_size)  # Sample actions (time x (batch x candidates) x actions)
			# Sample next states

			beliefs, states, _, _ = self.upper_transition_model(state, actions, belief)

			# if args.MultiGPU:
			#   actions_trans = torch.transpose(actions, 0, 1).cuda()
			#   beliefs, states, _, _ = self.transition_model(state, actions_trans, belief)
			#   beliefs, states = list(map(lambda x: x.view(-1, self.candidates, x.shape[2]), [beliefs, states]))
			#
			# else:
			#   beliefs, states, _, _ = self.transition_model(state, actions, belief)
			# beliefs, states, _, _ = self.transition_model(state, actions, belief)# [12, 1000, 200] [12, 1000, 30] : 12 horizon steps; 1000 candidates

			# Calculate expected returns (technically sum of rewards over planning horizon)
			returns = self.reward_model(beliefs.view(-1, H), states.view(-1, Z)).view(self.top_planning_horizon, -1).sum(dim=0)  # output from r-model[12000]->view[12, 1000]->sum[1000]
			# Re-fit belief to the K best action sequencessetting -> Repositories
			_, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
			topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)  # Fix indices for unrolled actions
			best_actions = actions[:, topk.view(-1)].reshape(self.top_planning_horizon, B, self.top_candidates, self.action_size)
			# Update belief with new means and standard deviations
			action_mean, action_std_dev = best_actions.mean(dim=2, keepdim=True), best_actions.std(dim=2, unbiased=False, keepdim=True)
		# Return first action mean µ_t
		return action_mean[0].squeeze(dim=1)


class Algorithms(MPCPlanner):

	def __init__(self, action_size, transition_model, encoder, reward_model, observation_model):
		super().__init__(action_size, args.top_planning_horizon, args.optimisation_iters, args.candidates, args.top_candidates)
		# self.planner = MPCPlanner(action_size, args.planning_horizon, args.optimisation_iters, args.candidates, args.top_candidates, transition_model, reward_model)
		self.encoder, self.reward_model, self.transition_model, self.observation_model = encoder, reward_model, transition_model, observation_model

		# self.merge_value_model = ValueModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(device=args.device)
		# self.merge_actor_model = MergeModel(args.belief_size, args.state_size, args.hidden_size, action_size, args.pool_len, args.dense_activation_function).to(device=args.device)
		# self.merge_actor_model.share_memory()
		# self.merge_value_model.share_memory()

		# set actor, value pool
		self.actor_pool = [ActorModel(args.belief_size, args.state_size, args.hidden_size, action_size, args.dense_activation_function).to(device=args.device) for _ in range(args.pool_len)]
		self.value_pool = [ValueModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(device=args.device) for _ in range(args.pool_len)]
		[actor.share_memory() for actor in self.actor_pool]
		[value.share_memory() for value in self.value_pool]

		# self.env_model_modules = get_modules([self.transition_model, self.encoder, self.observation_model, self.reward_model])
		# self.actor_pool_modules = get_modules(self.actor_pool)
		# self.model_modules = self.env_model_modules + self.actor_pool_modules

		# self.merge_value_model_modules = get_modules([self.merge_value_model])
		#
		# self.merge_actor_optimizer = optim.Adam(self.merge_actor_model.parameters(), lr=0 if args.learning_rate_schedule != 0 else args.actor_learning_rate, eps=args.adam_epsilon)
		# self.merge_value_optimizer = optim.Adam(self.merge_value_model.parameters(), lr=0 if args.learning_rate_schedule != 0 else args.value_learning_rate, eps=args.adam_epsilon)

		# self.actor_pool_0_optimizer = optim.Adam(self.actor_pool[0].parameters(), lr=0 if args.learning_rate_schedule != 0 else args.actor_learning_rate, eps=args.adam_epsilon)
		# self.value_pool_0_optimizer = opt


		self.actor_pipes = [Pipe() for i in range(1, len(self.actor_pool) + 1)]  # Set Multi Pipe
		self.workers_actor = [Worker_actor(actor_l=self.actor_pool[i],
										   value_l=self.value_pool[i],
										   transition_model=self.transition_model,
										   encoder=self.encoder,
										   observation_model=self.observation_model,
										   reward_model=self.reward_model,
										   child_conn=child,
										   results_dir=args.results_dir,
										   id=i + 1) for i, [parent, child] in enumerate(self.actor_pipes)]  # Set Worker_actor Using i'th actor_pipes

		[w.start() for i, w in enumerate(self.workers_actor)]  # Start Single Process

		self.metrics = {'episodes': [], 'merge_actor_loss': [], 'merge_value_loss': []}
		self.merge_losses = []

	def get_action_sequence(self, belief_init, state_init, B):
		actions_l_mean_lists, actions_l_std_lists = [], []
		for actor_l in self.actor_pool:
			actions_l_mean, actions_l_std = actor_l.get_action_mean_std(belief_init, state_init)
			actions_l_mean_list = actions_l_mean.unsqueeze(dim=1).expand(self.top_planning_horizon, B, 1, self.action_size)
			actions_l_std_list = actions_l_std.unsqueeze(dim=1).expand(self.top_planning_horizon, B, 1, self.action_size)
			actions_l_mean_lists.append(actions_l_mean_list)
			actions_l_std_lists.append(actions_l_std_list)
		return actions_l_mean_lists, actions_l_std_lists

	# def get_action(self, belief, posterior_state, explore=False, det=False):
	# 	action = self.actor_pool[0].get_action(belief, posterior_state, det=not (explore))
	# 	return action

	# def get_action2(self, belief, posterior_state, explore=False, det=False):
	# 	actions_l_mean, actions_l_std = self.actor_pool[0].get_action_mean_std(belief, posterior_state)
	# 	dist = Normal(actions_l_mean, actions_l_std)
	# 	dist = TransformedDistribution(dist, TanhBijector())
	# 	dist = torch.distributions.Independent(dist, 1)
	# 	dist = SampleDist(dist)
	# 	if det:
	# 		return dist.mode()
	# 	else:
	# 		return dist.rsample()

	def get_action(self, belief, posterior_state, explore=False, det=False):
		state = posterior_state
		B, H, Z = belief.size(0), belief.size(1), state.size(1)

		actions_l_mean_lists, actions_l_std_lists = self.get_action_sequence(belief, state, B)

		belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)

		# Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
		# action_mean, action_std_dev = torch.zeros(self.planning_horizon, B, 1, self.action_size, device=belief.device), torch.ones(self.planning_horizon, B, 1, self.action_size, device=belief.device)
		action_mean, action_std_dev = None, None
		for _ in range(self.optimisation_iters):
			# print("optimization_iters",_)
			# Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
			if _ == 0:
				sub_action_list = []
				for id in range(len(self.actor_pool)):
					# a = self.candidates//len(self.actor_pool)
					action = (actions_l_mean_lists[id] + actions_l_std_lists[id] * torch.randn(self.top_planning_horizon, B, self.candidates // len(self.actor_pool), self.action_size, device=belief.device)).view(self.top_planning_horizon,
																																																				B * self.candidates // len(
																																																					self.actor_pool),
																																																				self.action_size)  # Sample actions (time x (batch x candidates) x actions)
					sub_action_list.append(action)
				actions = torch.cat(sub_action_list, dim=1)
			else:
				actions = (action_mean + action_std_dev * torch.randn(self.top_planning_horizon, B, self.candidates, self.action_size, device=belief.device)).view(self.top_planning_horizon, B * self.candidates, self.action_size)  # Sample actions (time x (batch x candidates) x actions)
			# Sample next states

			beliefs, states, _, _ = self.upper_transition_model(state, actions, belief)

			# if args.MultiGPU:
			#   actions_trans = torch.transpose(actions, 0, 1).cuda()
			#   beliefs, states, _, _ = self.transition_model(state, actions_trans, belief)
			#   beliefs, states = list(map(lambda x: x.view(-1, self.candidates, x.shape[2]), [beliefs, states]))
			#
			# else:
			#   beliefs, states, _, _ = self.transition_model(state, actions, belief)
			# beliefs, states, _, _ = self.transition_model(state, actions, belief)# [12, 1000, 200] [12, 1000, 30] : 12 horizon steps; 1000 candidates

			# Calculate expected returns (technically sum of rewards over planning horizon)
			returns = self.reward_model(beliefs.view(-1, H), states.view(-1, Z)).view(self.top_planning_horizon, -1).sum(dim=0)  # output from r-model[12000]->view[12, 1000]->sum[1000]
			# Re-fit belief to the K best action sequencessetting -> Repositories
			_, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
			topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)  # Fix indices for unrolled actions
			best_actions = actions[:, topk.view(-1)].reshape(self.top_planning_horizon, B, self.top_candidates, self.action_size)
			# Update belief with new means and standard deviations
			action_mean, action_std_dev = best_actions.mean(dim=2, keepdim=True), best_actions.std(dim=2, unbiased=False, keepdim=True)

		# Return sample action from distribution

		dist = Normal(action_mean[0].squeeze(dim=1), action_std_dev[0].squeeze(dim=1))
		dist = TransformedDistribution(dist, TanhBijector())
		dist = torch.distributions.Independent(dist, 1)
		dist = SampleDist(dist)
		if det:
			tmp = dist.mode()
			return tmp
		else:
			tmp = dist.rsample()
			return tmp
		# action_true = action_mean[0].squeeze(dim=1)
		# return action_true

		# action = self.planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
		# return action

	def train_algorithm(self, actor_states, actor_beliefs):
		# "the planet no train step"
		[self.actor_pipes[i][0].send(1) for i, w in enumerate(self.workers_actor)]  # Parent_pipe send data using i'th pipes
		[self.actor_pipes[i][0].recv() for i, _ in enumerate(self.actor_pool)]  # waitting the children finish

	# def train_algorithm1(self, actor_states, actor_beliefs) -> None:
	# 	# print("children process {} waiting to get data".format(self.process_id))
	# 	# Run = self.child_conn.recv()
	# 	# print("children process {} Geted data form parent".format(self.process_id))
	# 	actor_states = torch.load(os.path.join(os.getcwd(), self.results_dir + '/actor_states.pt'))
	# 	actor_beliefs = torch.load(os.path.join(os.getcwd(), self.results_dir + '/actor_beliefs.pt'))
	#
	# 	with FreezeParameters(self.env_model_modules):
	#
	# 		actor_states = actor_states.cuda() if torch.cuda.is_available() and not args.disable_cuda else actor_states.cpu()
	# 		actor_beliefs = actor_beliefs.cuda() if torch.cuda.is_available() and not args.disable_cuda else actor_beliefs.cpu()
	#
	# 		imagination_traj = imagine_ahead(actor_states, actor_beliefs, self.actor_pool[0], self.transition_model, args.planning_horizon, action_scale=self.process_id)
	# 	imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs = imagination_traj
	#
	# 	# Update model parameters
	# 	with FreezeParameters(self.env_model_modules + self.value_pool[0]):
	# 		imged_reward = bottle(self.reward_model, (imged_beliefs, imged_prior_states))
	# 		value_pred = bottle(self.value_pool[0], (imged_beliefs, imged_prior_states))
	#
	# 	returns = lambda_return(imged_reward, value_pred, bootstrap=value_pred[-1], discount=args.discount, lambda_=args.disclam)
	# 	actor_loss = -torch.mean(returns)
	#
	# 	# calculate local gradients and push local parameters to global
	# 	self.actor_pool_0_optimizer.zero_grad()
	# 	actor_loss.backward()
	# 	nn.utils.clip_grad_norm_(self.actor_pool[0].parameters(), args.grad_clip_norm, norm_type=2)
	# 	# for la, ga in zip(self.actor_l.parameters(), self.actor_g.parameters()):
	# 	#     ga._grad = la.grad
	# 	self.actor_pool_0_optimizer.step()
	#
	# 	# push global parameters
	# 	# self.actor_l.load_state_dict(self.actor_g.state_dict())
	#
	# 	# Dreamer implementation: value loss calculation and optimization
	# 	with torch.no_grad():
	# 		value_beliefs = imged_beliefs.detach()
	# 		value_prior_states = imged_prior_states.detach()
	# 		target_return = returns.detach()
	#
	# 	value_dist = Normal(bottle(self.value_pool[0], (value_beliefs, value_prior_states)), 1)  # detach the input tensor from the transition network.
	# 	value_loss = -value_dist.log_prob(target_return).mean(dim=(0, 1))
	# 	# Update model parameters
	# 	self.value_l.zero_grad()
	# 	value_loss.backward()
	# 	nn.utils.clip_grad_norm_(self.value_l.parameters(), args.grad_clip_norm, norm_type=2)
	# 	self.value_optimizer_g.step()

	def save_loss_data(self, metrics_episodes):
		# losses = tuple(zip(*self.merge_losses))
		# self.metrics['merge_actor_loss'].append(losses[0])
		# self.metrics['merge_value_loss'].append(losses[1])
		# Save_Txt(metrics_episodes[-1], self.metrics['merge_actor_loss'][-1], 'merge_actor_loss', args.results_dir)
		# Save_Txt(metrics_episodes[-1], self.metrics['merge_value_loss'][-1], 'merge_value_loss', args.results_dir)
		[sub_actor.save_loss_data(metrics_episodes) for sub_actor in self.workers_actor]  # save sub actor loss
		# self.merge_losses = []
		# pass

	# "the loss no loss data"
	# return None

	def train_to_eval(self):
		[actor_model.eval() for actor_model in self.actor_pool]
		[value_model.eval() for value_model in self.value_pool]
		# self.merge_actor_model.eval()
		# self.merge_value_model.eval()

	# return None
	def eval_to_train(self):
		[actor_model.train() for actor_model in self.actor_pool]
		[value_model.train() for value_model in self.value_pool]
		# self.merge_actor_model.train()
		# self.merge_value_model.train()
	# "the planet no eval_to_train step"
	# return None

def atanh(x):
	return 0.5 * torch.log((1 + x) / (1 - x))


class TanhBijector(torch.distributions.Transform):
	domain: constraints.Constraint
	codomain: constraints.Constraint

	def __init__(self):
		super().__init__()
		self.bijective = True
		self.domain = constraints.Constraint()
		self.codomain = constraints.Constraint()

	@property
	def sign(self): return 1.

	def _call(self, x): return torch.tanh(x)

	def _inverse(self, y: torch.Tensor):
		y = torch.where(
			(torch.abs(y) <= 1.),
			torch.clamp(y, -0.99999997, 0.99999997),
			y)
		y = atanh(y)
		return y

	def log_abs_det_jacobian(self, x, y):
		return 2. * (np.log(2) - x - F.softplus(-2. * x))

class SampleDist:
	def __init__(self, dist, samples=100):
		self._dist = dist
		self._samples = samples

	@property
	def name(self):
		return 'SampleDist'

	def __getattr__(self, name):
		return getattr(self._dist, name)

	def mean(self):
		sample = dist.rsample()
		return torch.mean(sample, 0)

	def mode(self):
		dist = self._dist.expand((self._samples, *self._dist.batch_shape))
		sample = dist.rsample()
		logprob = dist.log_prob(sample)
		batch_size = sample.size(1)
		feature_size = sample.size(2)
		indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
		return torch.gather(sample, 0, indices).squeeze(0)

	def entropy(self):
		dist = self._dist.expand((self._samples, *self._dist.batch_shape))
		sample = dist.rsample()
		logprob = dist.log_prob(sample)
		return -torch.mean(logprob, 0)

	def sample(self):
		return self._dist.sample()