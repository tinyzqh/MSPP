import torch
from torch import nn, optim
from torch.nn import functional as F
from parameter import args
from models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel, ValueModel, ActorModel, MergeModel
from utils import lineplot, write_video, imagine_ahead, lambda_return, FreezeParameters, Save_Txt, ActivateParameters, get_modules
from torch.distributions import Normal
from asynchronous_init_sample import Worker_init_Sample
from torch.multiprocessing import Pipe, Manager
from asynchronous_actor import Worker_actor


class Algorithms(object):

	def __init__(self, action_size, transition_model, encoder, reward_model, observation_model):

		self.encoder, self.reward_model, self.transition_model, self.observation_model = encoder, reward_model, transition_model, observation_model

		self.merge_value_model = ValueModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(device=args.device)
		self.merge_actor_model = MergeModel(args.belief_size, args.state_size, args.hidden_size, action_size, args.pool_len, args.dense_activation_function).to(device=args.device)
		self.merge_actor_model.share_memory()
		self.merge_value_model.share_memory()

		# set actor, value pool
		self.actor_pool = [ActorModel(args.belief_size, args.state_size, args.hidden_size, action_size, args.dense_activation_function).to(device=args.device) for _ in range(args.pool_len)]
		self.value_pool = [ValueModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(device=args.device) for _ in range(args.pool_len)]
		[actor.share_memory() for actor in self.actor_pool]
		[value.share_memory() for value in self.value_pool]

		self.env_model_modules = get_modules([self.transition_model, self.encoder, self.observation_model, self.reward_model])
		self.actor_pool_modules = get_modules(self.actor_pool)
		self.model_modules = self.env_model_modules + self.actor_pool_modules

		self.merge_value_model_modules = get_modules([self.merge_value_model])

		self.merge_actor_optimizer = optim.Adam(self.merge_actor_model.parameters(), lr=0 if args.learning_rate_schedule != 0 else args.actor_learning_rate, eps=args.adam_epsilon)
		self.merge_value_optimizer = optim.Adam(self.merge_value_model.parameters(), lr=0 if args.learning_rate_schedule != 0 else args.value_learning_rate, eps=args.adam_epsilon)

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

	def get_action(self, belief, posterior_state, explore=False):
		merge_action_list = []
		for actor_l in self.actor_pool:
			actions_l_mean, actions_l_std = actor_l.get_action_mean_std(belief, posterior_state)
			merge_action_list.append(actions_l_mean)
			merge_action_list.append(actions_l_std)
		merge_actions = torch.cat(merge_action_list, dim=1)
		action = self.merge_actor_model.get_merge_action(merge_actions, belief, posterior_state, det=not (explore))
		return action

	def train_algorithm(self, actor_states, actor_beliefs):

		[self.actor_pipes[i][0].send(1) for i, w in enumerate(self.workers_actor)]  # Parent_pipe send data using i'th pipes
		[self.actor_pipes[i][0].recv() for i, _ in enumerate(self.actor_pool)]  # waitting the children finish

		with FreezeParameters(self.model_modules):
			imagination_traj = self.imagine_merge_ahead(prev_state=actor_states, prev_belief=actor_beliefs, policy_pool=self.actor_pool, transition_model=self.transition_model, merge_model=self.merge_actor_model)
		imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs = imagination_traj

		with FreezeParameters(self.model_modules + self.merge_value_model_modules):
			imged_reward = bottle(self.reward_model, (imged_beliefs, imged_prior_states))
			value_pred = bottle(self.merge_value_model, (imged_beliefs, imged_prior_states))

		with FreezeParameters(self.actor_pool_modules):
			returns = lambda_return(imged_reward, value_pred, bootstrap=value_pred[-1], discount=args.discount, lambda_=args.disclam)
			merge_actor_loss = -torch.mean(returns)
			# Update model parameters
			self.merge_actor_optimizer.zero_grad()
			merge_actor_loss.backward()
			nn.utils.clip_grad_norm_(self.merge_actor_model.parameters(), args.grad_clip_norm, norm_type=2)
			self.merge_actor_optimizer.step()

		# Dreamer implementation: value loss calculation and optimization
		with torch.no_grad():
			value_beliefs = imged_beliefs.detach()
			value_prior_states = imged_prior_states.detach()
			target_return = returns.detach()

		value_dist = Normal(bottle(self.merge_value_model, (value_beliefs, value_prior_states)), 1)  # detach the input tensor from the transition network.
		merge_value_loss = -value_dist.log_prob(target_return).mean(dim=(0, 1))
		# Update model parameters
		self.merge_value_optimizer.zero_grad()
		merge_value_loss.backward()
		nn.utils.clip_grad_norm_(self.merge_value_model.parameters(), args.grad_clip_norm, norm_type=2)
		self.merge_value_optimizer.step()

		self.merge_losses.append([merge_actor_loss.item(), merge_value_loss.item()])

		# return [merge_actor_loss, merge_value_loss]

	def save_loss_data(self, metrics_episodes):
		losses = tuple(zip(*self.merge_losses))
		self.metrics['merge_actor_loss'].append(losses[0])
		self.metrics['merge_value_loss'].append(losses[1])
		Save_Txt(metrics_episodes[-1], self.metrics['merge_actor_loss'][-1], 'merge_actor_loss', args.results_dir)
		Save_Txt(metrics_episodes[-1], self.metrics['merge_value_loss'][-1], 'merge_value_loss', args.results_dir)
		[sub_actor.save_loss_data(metrics_episodes) for sub_actor in self.workers_actor]  # save sub actor loss
		self.merge_losses = []

	def imagine_merge_ahead(self, prev_state, prev_belief, policy_pool, transition_model, merge_model, planning_horizon=12):
		flatten = lambda x: x.view([-1] + list(x.size()[2:]))
		prev_belief = flatten(prev_belief)
		prev_state = flatten(prev_state)

		# Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
		T = planning_horizon
		beliefs, prior_states, prior_means, prior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [
			torch.empty(0)] * T, [torch.empty(0)] * T
		beliefs[0], prior_states[0] = prev_belief, prev_state
		for t in range(T - 1):
			_state = prior_states[t]

			merge_action_list = []
			for actor_l in policy_pool:
				actions_l_mean, actions_l_std = actor_l.get_action_mean_std(beliefs[t].detach(), _state.detach())
				merge_action_list.append(actions_l_mean)
				merge_action_list.append(actions_l_std)

			merge_actions = torch.cat(merge_action_list, dim=1)

			actions = merge_model.get_merge_action(merge_actions, beliefs[t].detach(), _state.detach())
			# Compute belief (deterministic hidden state)
			if args.MultiGPU and torch.cuda.device_count() > 1:
				hidden = transition_model.module.act_fn(transition_model.module.fc_embed_state_action(torch.cat([_state, actions], dim=1)))
				beliefs[t + 1] = transition_model.module.rnn(hidden, beliefs[t])
				# Compute state prior by applying transition dynamics
				hidden = transition_model.module.act_fn(transition_model.module.fc_embed_belief_prior(beliefs[t + 1]))
				prior_means[t + 1], _prior_std_dev = torch.chunk(transition_model.module.fc_state_prior(hidden), 2, dim=1)
				prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + transition_model.module.min_std_dev
			else:
				hidden = transition_model.act_fn(transition_model.fc_embed_state_action(torch.cat([_state, actions], dim=1)))
				beliefs[t + 1] = transition_model.rnn(hidden, beliefs[t])
				# Compute state prior by applying transition dynamics
				hidden = transition_model.act_fn(transition_model.fc_embed_belief_prior(beliefs[t + 1]))
				prior_means[t + 1], _prior_std_dev = torch.chunk(transition_model.fc_state_prior(hidden), 2, dim=1)
				prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + transition_model.min_std_dev
			prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
			# Return new hidden states
		# imagined_traj = [beliefs, prior_states, prior_means, prior_std_devs]
		imagined_traj = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
		return imagined_traj

	def train_to_eval(self):
		[actor_model.eval() for actor_model in self.actor_pool]
		[value_model.eval() for value_model in self.value_pool]
		self.merge_actor_model.eval()
		self.merge_value_model.eval()

	def eval_to_train(self):
		[actor_model.train() for actor_model in self.actor_pool]
		[value_model.train() for value_model in self.value_pool]
		self.merge_actor_model.train()
		self.merge_value_model.train()
