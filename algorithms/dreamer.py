import torch
import os
from torch import nn, optim
from torch.nn import functional as F
from parameter import args
from models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel, ValueModel, ActorModel, MergeModel
from utils import lineplot, write_video, imagine_ahead, lambda_return, FreezeParameters, Save_Txt, ActivateParameters, get_modules
from torch.distributions import Normal


class Algorithms(object):

	def __init__(self, action_size, transition_model, encoder, reward_model, observation_model):
		self.encoder, self.reward_model, self.transition_model, self.observation_model = encoder, reward_model, transition_model, observation_model

		self.actor_model = ActorModel(args.belief_size, args.state_size, args.hidden_size, action_size, args.dense_activation_function).to(device=args.device)
		self.value_model = ValueModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(device=args.device)
		self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=0 if args.learning_rate_schedule != 0 else args.actor_learning_rate, eps=args.adam_epsilon)
		self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=0 if args.learning_rate_schedule != 0 else args.value_learning_rate, eps=args.adam_epsilon)

		self.env_model_modules = get_modules([self.transition_model, self.encoder, self.observation_model, self.reward_model])
		self.value_model_modules = get_modules([self.value_model])

		self.sub_actor_losses = []
		self.metrics = {'episodes': [], 'actor_loss': [], 'value_loss': []}

	def get_action(self, belief, posterior_state, explore=False):
		action = self.actor_model.get_action(belief, posterior_state, det=not (explore))
		return action

	def train_algorithm(self, actor_states, actor_beliefs):


		actor_states = torch.load(os.path.join(os.getcwd(), args.results_dir + '/actor_states.pt'))
		actor_beliefs = torch.load(os.path.join(os.getcwd(), args.results_dir + '/actor_beliefs.pt'))

		with FreezeParameters(self.env_model_modules):
			imagination_traj = imagine_ahead(actor_states, actor_beliefs, self.actor_model, self.transition_model, args.planning_horizon, args.action_scale)
		imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs = imagination_traj

		with FreezeParameters(self.env_model_modules + self.value_model_modules):
			imged_reward = bottle(self.reward_model, (imged_beliefs, imged_prior_states))
			value_pred = bottle(self.value_model, (imged_beliefs, imged_prior_states))

		returns = lambda_return(imged_reward, value_pred, bootstrap=value_pred[-1], discount=args.discount, lambda_=args.disclam)
		actor_loss = -torch.mean(returns)

		# Update model parameters
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		nn.utils.clip_grad_norm_(self.actor_model.parameters(), args.grad_clip_norm, norm_type=2)
		self.actor_optimizer.step()

		# Dreamer implementation: value loss calculation and optimization
		with torch.no_grad():
			value_beliefs = imged_beliefs.detach()
			value_prior_states = imged_prior_states.detach()
			target_return = returns.detach()

		value_dist = Normal(bottle(self.value_model, (value_beliefs, value_prior_states)), 1)  # detach the input tensor from the transition network.
		value_loss = -value_dist.log_prob(target_return).mean(dim=(0, 1))

		# Update model parameters
		self.value_optimizer.zero_grad()
		value_loss.backward()
		nn.utils.clip_grad_norm_(self.value_model.parameters(), args.grad_clip_norm, norm_type=2)
		self.value_optimizer.step()

		self.sub_actor_losses.append([actor_loss.item(), value_loss.item()])
		# return [actor_loss, value_loss]

	def save_loss_data(self, metrics_episodes):
		losses = tuple(zip(*self.sub_actor_losses))
		self.metrics['actor_loss'].append(losses[0])
		self.metrics['value_loss'].append(losses[1])
		Save_Txt(metrics_episodes[-1], self.metrics['actor_loss'][-1], 'actor_loss', args.results_dir)
		Save_Txt(metrics_episodes[-1], self.metrics['value_loss'][-1], 'value_loss', args.results_dir)
		self.sub_actor_losses = []

	def train_to_eval(self):
		self.actor_model.eval()
		self.value_model.eval()

	def eval_to_train(self):
		self.actor_model.train()
		self.value_model.train()
