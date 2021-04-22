import os
import time
import sys
import numpy as np
import torch
import multiprocessing as mp
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher
from memory import ExperienceReplay
from models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel, ValueModel, ActorModel, MergeModel
from utils import lineplot, write_video, imagine_ahead, lambda_return, FreezeParameters, Save_Txt, ActivateParameters
from tensorboardX import SummaryWriter
from parameter import args
from asynchronous_init_sample import Worker_init_Sample
from torch.multiprocessing import Pipe, Manager


class Plan(object):

	def __init__(self):

		self.results_dir = os.path.join('results', '{}_seed_{}_{}_action_scale_{}_no_explore_{}_pool_len_{}_optimisation_iters_{}_top_planning-horizon'.format(args.env, args.seed, args.algo, args.action_scale, args.pool_len, args.optimisation_iters, args.top_planning_horizon))

		args.results_dir = self.results_dir
		args.MultiGPU = True if torch.cuda.device_count() > 1 and args.MultiGPU else False

		self.__basic_setting()
		self.__init_sample()  # Sampleing The Init Data

		# Initialise model parameters randomly
		self.transition_model = TransitionModel(args.belief_size, args.state_size, self.env.action_size, args.hidden_size, args.embedding_size, args.dense_activation_function).to(device=args.device)
		self.observation_model = ObservationModel(args.symbolic_env, self.env.observation_size, args.belief_size, args.state_size, args.embedding_size, args.cnn_activation_function).to(device=args.device)
		self.reward_model = RewardModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(device=args.device)
		self.encoder = Encoder(args.symbolic_env, self.env.observation_size, args.embedding_size, args.cnn_activation_function).to(device=args.device)

		print("We Have {} GPUS".format(torch.cuda.device_count())) if args.MultiGPU else print("We use CPU")
		self.transition_model = nn.DataParallel(self.transition_model.to(device=args.device)) if args.MultiGPU else self.transition_model
		self.observation_model = nn.DataParallel(self.observation_model.to(device=args.device)) if args.MultiGPU else self.observation_model
		self.reward_model = nn.DataParallel(self.reward_model.to(device=args.device)) if args.MultiGPU else self.reward_model

		# encoder = nn.DataParallel(encoder.cuda())
		# actor_model = nn.DataParallel(actor_model.cuda())
		# value_model = nn.DataParallel(value_model.cuda())

		# share the global parameters in multiprocessing
		self.encoder.share_memory()
		self.observation_model.share_memory()
		self.reward_model.share_memory()

		# Set all_model/global_actor_optimizer/global_value_optimizer
		self.param_list = list(self.transition_model.parameters()) + list(self.observation_model.parameters()) + list(self.reward_model.parameters()) + list(self.encoder.parameters())
		self.model_optimizer = optim.Adam(self.param_list, lr=0 if args.learning_rate_schedule != 0 else args.model_learning_rate, eps=args.adam_epsilon)

	def update_belief_and_act(self, args, env, belief, posterior_state, action, observation, explore=False):
		# Infer belief over current state q(s_t|o≤t,a<t) from the history
		# print("action size: ",action.size()) torch.Size([1, 6])
		belief, _, _, _, posterior_state, _, _ = self.upper_transition_model(posterior_state, action.unsqueeze(dim=0), belief, self.encoder(observation).unsqueeze(dim=0), None)
		if hasattr(env, "envs"): belief, posterior_state = list(map(lambda x: x.view(-1, args.test_episodes, x.shape[2]), [x for x in [belief, posterior_state]]))

		belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)  # Remove time dimension from belief/state
		action = self.algorithms.get_action(belief, posterior_state, explore)

		if explore:
			action = torch.clamp(Normal(action, args.action_noise).rsample(), -1, 1)  # Add gaussian exploration noise on top of the sampled action
			# action = action + args.action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
		next_observation, reward, done = env.step(action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu())  # Perform environment step (action repeats handled internally)
		return belief, posterior_state, action, next_observation, reward, done

	def run(self):
		if args.algo == "dreamer":
			print("DREAMER")
			from algorithms.dreamer import Algorithms
			self.algorithms = Algorithms(self.env.action_size, self.transition_model, self.encoder, self.reward_model, self.observation_model)
		elif args.algo == "p2p":
			print("planing to plan")
			from algorithms.plan_to_plan import Algorithms
			self.algorithms = Algorithms(self.env.action_size, self.transition_model, self.encoder, self.reward_model, self.observation_model)
		elif args.algo == "actor_pool_1":
			print("async sub actor")
			from algorithms.actor_pool_1 import Algorithms_actor
			self.algorithms = Algorithms_actor(self.env.action_size, self.transition_model, self.encoder, self.reward_model, self.observation_model)
		elif args.algo == "aap":
			from algorithms.asynchronous_actor_planet import Algorithms
			self.algorithms = Algorithms(self.env.action_size, self.transition_model, self.encoder, self.reward_model, self.observation_model)
		else:
			print("planet")
			from algorithms.planet import Algorithms
			# args.MultiGPU = False
			self.algorithms = Algorithms(self.env.action_size, self.transition_model, self.reward_model)

		if args.test: self.test_only()

		self.global_prior = Normal(torch.zeros(args.batch_size, args.state_size, device=args.device), torch.ones(args.batch_size, args.state_size, device=args.device))  # Global prior N(0, I)
		self.free_nats = torch.full((1,), args.free_nats, device=args.device)  # Allowed deviation in KL divergence

		# Training (and testing)
		# args.episodes = 1
		for episode in tqdm(range(self.metrics['episodes'][-1] + 1, args.episodes + 1), total=args.episodes, initial=self.metrics['episodes'][-1] + 1):
			losses = self.train()
			# self.algorithms.save_loss_data(self.metrics['episodes']) # Update and plot loss metrics
			self.save_loss_data(tuple(zip(*losses)))  # Update and plot loss metrics
			self.data_collection(episode=episode)  # Data collection
			# args.test_interval = 1
			if episode % args.test_interval == 0: self.test(episode=episode)  # Test model
			self.save_model_data(episode=episode)  # save model

		self.env.close()  # Close training environment

	def train_env_model(self, beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs, observations, actions, rewards, nonterminals):
		# Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
		if args.worldmodel_LogProbLoss:
			observation_dist = Normal(bottle(self.observation_model, (beliefs, posterior_states)), 1)
			observation_loss = -observation_dist.log_prob(observations[1:]).sum(dim=2 if args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))
		else:
			observation_loss = F.mse_loss(bottle(self.observation_model, (beliefs, posterior_states)), observations[1:], reduction='none').sum(dim=2 if args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))
		if args.worldmodel_LogProbLoss:
			reward_dist = Normal(bottle(self.reward_model, (beliefs, posterior_states)), 1)
			reward_loss = -reward_dist.log_prob(rewards[:-1]).mean(dim=(0, 1))
		else:
			reward_loss = F.mse_loss(bottle(self.reward_model, (beliefs, posterior_states)), rewards[:-1], reduction='none').mean(dim=(0, 1))

		# transition loss
		div = kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2)
		kl_loss = torch.max(div, self.free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
		if args.global_kl_beta != 0:
			kl_loss += args.global_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs), self.global_prior).sum(dim=2).mean(dim=(0, 1))
		# Calculate latent overshooting objective for t > 0
		if args.overshooting_kl_beta != 0:
			overshooting_vars = []  # Collect variables for overshooting to process in batch
			for t in range(1, args.chunk_size - 1):
				d = min(t + args.overshooting_distance, args.chunk_size - 1)  # Overshooting distance
				t_, d_ = t - 1, d - 1  # Use t_ and d_ to deal with different time indexing for latent states
				seq_pad = (0, 0, 0, 0, 0, t - d + args.overshooting_distance)  # Calculate sequence padding so overshooting terms can be calculated in one batch
				# Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
				overshooting_vars.append((F.pad(actions[t:d], seq_pad), F.pad(nonterminals[t:d], seq_pad),
										  F.pad(rewards[t:d], seq_pad[2:]), beliefs[t_], prior_states[t_],
										  F.pad(posterior_means[t_ + 1:d_ + 1].detach(), seq_pad),
										  F.pad(posterior_std_devs[t_ + 1:d_ + 1].detach(), seq_pad, value=1),
										  F.pad(torch.ones(d - t, args.batch_size, args.state_size, device=args.device),
												seq_pad)))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
			overshooting_vars = tuple(zip(*overshooting_vars))
			# Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
			beliefs, prior_states, prior_means, prior_std_devs = self.upper_transition_model(torch.cat(overshooting_vars[4], dim=0),
																							 torch.cat(overshooting_vars[0], dim=1),
																							 torch.cat(overshooting_vars[3], dim=0),
																							 None,
																							 torch.cat(overshooting_vars[1], dim=1))
			seq_mask = torch.cat(overshooting_vars[7], dim=1)
			# Calculate overshooting KL loss with sequence mask
			kl_loss += (1 / args.overshooting_distance) * args.overshooting_kl_beta * torch.max((kl_divergence(
				Normal(torch.cat(overshooting_vars[5], dim=1), torch.cat(overshooting_vars[6], dim=1)),
				Normal(prior_means, prior_std_devs)) * seq_mask).sum(dim=2), self.free_nats).mean(dim=(0, 1)) * (
							   args.chunk_size - 1)  # Update KL loss (compensating for extra average over each overshooting/open loop sequence)
			# Calculate overshooting reward prediction loss with sequence mask
			if args.overshooting_reward_scale != 0:
				reward_loss += (1 / args.overshooting_distance) * args.overshooting_reward_scale * F.mse_loss(bottle(self.reward_model, (beliefs, prior_states)) * seq_mask[:, :, 0], torch.cat(overshooting_vars[2], dim=1),reduction='none').mean(dim=(0, 1)) * (args.chunk_size - 1)  # Update reward loss (compensating for extra average over each overshooting/open loop sequence)
		# Apply linearly ramping learning rate schedule
		if args.learning_rate_schedule != 0:
			for group in self.model_optimizer.param_groups:
				group['lr'] = min(group['lr'] + args.model_learning_rate / args.model_learning_rate_schedule,
								  args.model_learning_rate)
		model_loss = observation_loss + reward_loss + kl_loss
		# Update model parameters
		self.model_optimizer.zero_grad()
		model_loss.backward()
		nn.utils.clip_grad_norm_(self.param_list, args.grad_clip_norm, norm_type=2)
		self.model_optimizer.step()
		return observation_loss, reward_loss, kl_loss

	def train(self):
		# Model fitting
		losses = []
		print("training loop")
		# args.collect_interval = 1
		for s in tqdm(range(args.collect_interval)):

			# Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
			observations, actions, rewards, nonterminals = self.D.sample(args.batch_size, args.chunk_size)  # Transitions start at time t = 0
			# Create initial belief and state for time t = 0
			init_belief, init_state = torch.zeros(args.batch_size, args.belief_size, device=args.device), torch.zeros(args.batch_size, args.state_size, device=args.device)
			# Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
			obs = bottle(self.encoder, (observations[1:],))
			beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = self.upper_transition_model(prev_state=init_state, actions=actions[:-1], prev_belief=init_belief, obs=obs,
																																					nonterminals=nonterminals[:-1])

			# Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
			observation_loss, reward_loss, kl_loss = self.train_env_model(beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs, observations, actions, rewards, nonterminals)

			# Dreamer implementation: actor loss calculation and optimization
			with torch.no_grad():
				actor_states = posterior_states.detach().to(device=args.device).share_memory_()
				actor_beliefs = beliefs.detach().to(device=args.device).share_memory_()


			# if not os.path.exists(os.path.join(os.getcwd(), 'tensor_data/' + args.results_dir)): os.mkdir(os.path.join(os.getcwd(), 'tensor_data/' + args.results_dir))
			torch.save(actor_states, os.path.join(os.getcwd(), args.results_dir + '/actor_states.pt'))
			torch.save(actor_beliefs, os.path.join(os.getcwd(), args.results_dir + '/actor_beliefs.pt'))

			# [self.actor_pipes[i][0].send(1) for i, w in enumerate(self.workers_actor)]  # Parent_pipe send data using i'th pipes
			# [self.actor_pipes[i][0].recv() for i, _ in enumerate(self.actor_pool)]  # waitting the children finish

			self.algorithms.train_algorithm(actor_states, actor_beliefs)
			losses.append([observation_loss.item(), reward_loss.item(), kl_loss.item()])

			# if self.algorithms.train_algorithm(actor_states, actor_beliefs) is not None:
			#   merge_actor_loss, merge_value_loss = self.algorithms.train_algorithm(actor_states, actor_beliefs)
			#   losses.append([observation_loss.item(), reward_loss.item(), kl_loss.item(), merge_actor_loss.item(), merge_value_loss.item()])
			# else:
			#   losses.append([observation_loss.item(), reward_loss.item(), kl_loss.item()])

		return losses

	def data_collection(self, episode):
		print("Data collection")
		with torch.no_grad():
			observation, total_reward = self.env.reset(), 0
			belief, posterior_state, action = torch.zeros(1, args.belief_size, device=args.device), torch.zeros(1, args.state_size, device=args.device), torch.zeros(1, self.env.action_size, device=args.device)
			pbar = tqdm(range(args.max_episode_length // args.action_repeat))
			for t in pbar:
				# print("step",t)
				belief, posterior_state, action, next_observation, reward, done = self.update_belief_and_act(args, self.env, belief, posterior_state, action, observation.to(device=args.device))
				self.D.append(observation, action.cpu(), reward, done)
				total_reward += reward
				observation = next_observation
				if args.render: self.env.render()
				if done:
					pbar.close()
					break

			# Update and plot train reward metrics
			self.metrics['steps'].append(t + self.metrics['steps'][-1])
			self.metrics['episodes'].append(episode)
			self.metrics['train_rewards'].append(total_reward)

			Save_Txt(self.metrics['episodes'][-1], self.metrics['train_rewards'][-1], 'train_rewards', args.results_dir)
			# lineplot(metrics['episodes'][-len(metrics['train_rewards']):], metrics['train_rewards'], 'train_rewards', results_dir)

	def test(self, episode):
		print("Test model")
		# Set models to eval mode
		self.transition_model.eval()
		self.observation_model.eval()
		self.reward_model.eval()
		self.encoder.eval()
		self.algorithms.train_to_eval()
		# self.actor_model_g.eval()
		# self.value_model_g.eval()
		# Initialise parallelised test environments
		test_envs = EnvBatcher(Env, (args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth), {}, args.test_episodes)

		with torch.no_grad():
			observation, total_rewards, video_frames = test_envs.reset(), np.zeros((args.test_episodes,)), []
			belief, posterior_state, action = torch.zeros(args.test_episodes, args.belief_size, device=args.device), torch.zeros(args.test_episodes, args.state_size, device=args.device), torch.zeros(args.test_episodes, self.env.action_size,
																																																	   device=args.device)
			pbar = tqdm(range(args.max_episode_length // args.action_repeat))
			for t in pbar:
				belief, posterior_state, action, next_observation, reward, done = self.update_belief_and_act(args, test_envs, belief, posterior_state, action, observation.to(device=args.device))
				total_rewards += reward.numpy()
				if not args.symbolic_env:  # Collect real vs. predicted frames for video
					video_frames.append(make_grid(torch.cat([observation, self.observation_model(belief, posterior_state).cpu()], dim=3) + 0.5, nrow=5).numpy())  # Decentre
				observation = next_observation
				if done.sum().item() == args.test_episodes:
					pbar.close()
					break

		# Update and plot reward metrics (and write video if applicable) and save metrics
		self.metrics['test_episodes'].append(episode)
		self.metrics['test_rewards'].append(total_rewards.tolist())

		Save_Txt(self.metrics['test_episodes'][-1], self.metrics['test_rewards'][-1], 'test_rewards', args.results_dir)
		# Save_Txt(np.asarray(metrics['steps'])[np.asarray(metrics['test_episodes']) - 1], metrics['test_rewards'],'test_rewards_steps', results_dir, xaxis='step')

		# lineplot(metrics['test_episodes'], metrics['test_rewards'], 'test_rewards', results_dir)
		# lineplot(np.asarray(metrics['steps'])[np.asarray(metrics['test_episodes']) - 1], metrics['test_rewards'], 'test_rewards_steps', results_dir, xaxis='step')
		if not args.symbolic_env:
			episode_str = str(episode).zfill(len(str(args.episodes)))
			write_video(video_frames, 'test_episode_%s' % episode_str, args.results_dir)  # Lossy compression
			save_image(torch.as_tensor(video_frames[-1]), os.path.join(args.results_dir, 'test_episode_%s.png' % episode_str))

		torch.save(self.metrics, os.path.join(args.results_dir, 'metrics.pth'))

		# Set models to train mode
		self.transition_model.train()
		self.observation_model.train()
		self.reward_model.train()
		self.encoder.train()
		# self.actor_model_g.train()
		# self.value_model_g.train()
		self.algorithms.eval_to_train()
		# Close test environments
		test_envs.close()

	def test_only(self):
		# Set models to eval mode
		self.transition_model.eval()
		self.reward_model.eval()
		self.encoder.eval()
		with torch.no_grad():
			total_reward = 0
			for _ in tqdm(range(args.test_episodes)):
				observation = self.env.reset()
				belief, posterior_state, action = torch.zeros(1, args.belief_size, device=args.device), torch.zeros(1, args.state_size, device=args.device), torch.zeros(1, self.env.action_size, device=args.device)
				pbar = tqdm(range(args.max_episode_length // args.action_repeat))
				for t in pbar:
					belief, posterior_state, action, observation, reward, done = self.update_belief_and_act(args, self.env, belief, posterior_state, action, observation.to(evice=args.device))
					total_reward += reward
					if args.render: self.env.render()
					if done:
						pbar.close()
						break
		print('Average Reward:', total_reward / args.test_episodes)
		self.env.close()
		quit()

	def __basic_setting(self):
		args.overshooting_distance = min(args.chunk_size, args.overshooting_distance)  # Overshooting distance cannot be greater than chunk size
		print(' ' * 26 + 'Options')
		for k, v in vars(args).items():
			print(' ' * 26 + k + ': ' + str(v))

		print("torch.cuda.device_count() {}".format(torch.cuda.device_count()))
		os.makedirs(args.results_dir, exist_ok=True)
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		# Set Cuda
		if torch.cuda.is_available() and not args.disable_cuda:
			print("using CUDA")
			args.device = torch.device('cuda')
			torch.cuda.manual_seed(args.seed)
		else:
			print("using CPU")
			args.device = torch.device('cpu')

		self.summary_name = args.results_dir + "/{}_{}_log"
		self.writer = SummaryWriter(self.summary_name.format(args.env, args.id))
		self.env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth)
		self.metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [], 'observation_loss': [], 'reward_loss': [], 'kl_loss': [], 'merge_actor_loss': [], 'merge_value_loss': []}

	def __init_sample(self):
		if args.experience_replay is not '' and os.path.exists(args.experience_replay):
			self.D = torch.load(args.experience_replay)
			self.metrics['steps'], self.metrics['episodes'] = [self.D.steps] * self.D.episodes, list(range(1, self.D.episodes + 1))
		elif not args.test:
			self.D = ExperienceReplay(args.experience_size, args.symbolic_env, self.env.observation_size, self.env.action_size, args.bit_depth, args.device)

			# Initialise dataset D with S random seed episodes
			print("Start Multi Sample Processing -------------------------------")
			start_time = time.time()
			data_lists = [Manager().list() for i in range(1, args.seed_episodes + 1)]  # Set Global Lists
			pipes = [Pipe() for i in range(1, args.seed_episodes + 1)]  # Set Multi Pipe
			workers_init_sample = [Worker_init_Sample(child_conn=child, id=i + 1) for i, [parent, child] in enumerate(pipes)]

			for i, w in enumerate(workers_init_sample):
				w.start()  # Start Single Process
				pipes[i][0].send(data_lists[i])  # Parent_pipe send data using i'th pipes
			[w.join() for w in workers_init_sample]  # wait sub_process done

			for i, [parent, child] in enumerate(pipes):
				# datas = parent.recv()
				for data in list(parent.recv()):
					if isinstance(data, tuple):
						assert len(data) == 4
						self.D.append(data[0], data[1], data[2], data[3])
					elif isinstance(data, int):
						t = data
						self.metrics['steps'].append(t * args.action_repeat + (0 if len(self.metrics['steps']) == 0 else self.metrics['steps'][-1]))
						self.metrics['episodes'].append(i + 1)
					else:
						print("The Recvive Data Have Some Problems, Need To Fix")
			end_time = time.time()
			print("the process times {} s".format(end_time - start_time))
			print("End Multi Sample Processing -------------------------------")

	def upper_transition_model(self, prev_state, actions, prev_belief, obs, nonterminals):
		actions = torch.transpose(actions, 0, 1) if args.MultiGPU else actions
		nonterminals = torch.transpose(nonterminals, 0, 1).to(device=args.device) if args.MultiGPU and nonterminals is not None else nonterminals
		obs = torch.transpose(obs, 0, 1).to(device=args.device) if args.MultiGPU and obs is not None else obs
		temp_val = self.transition_model(prev_state.to(device=args.device), actions.to(device=args.device), prev_belief.to(device=args.device), obs, nonterminals)

		return list(map(lambda x: torch.cat(x.chunk(torch.cuda.device_count(), 0), 1) if x.shape[1] != prev_state.shape[0] else x, [x for x in temp_val]))

	def save_loss_data(self, losses):
		self.metrics['observation_loss'].append(losses[0])
		self.metrics['reward_loss'].append(losses[1])
		self.metrics['kl_loss'].append(losses[2])
		self.metrics['merge_actor_loss'].append(losses[3]) if losses.__len__() > 3 else None
		self.metrics['merge_value_loss'].append(losses[4]) if losses.__len__() > 3 else None

		Save_Txt(self.metrics['episodes'][-1], self.metrics['observation_loss'][-1], 'observation_loss', args.results_dir)
		Save_Txt(self.metrics['episodes'][-1], self.metrics['reward_loss'][-1], 'reward_loss', args.results_dir)
		Save_Txt(self.metrics['episodes'][-1], self.metrics['kl_loss'][-1], 'kl_loss', args.results_dir)
		Save_Txt(self.metrics['episodes'][-1], self.metrics['merge_actor_loss'][-1], 'merge_actor_loss', args.results_dir) if losses.__len__() > 3 else None
		Save_Txt(self.metrics['episodes'][-1], self.metrics['merge_value_loss'][-1], 'merge_value_loss', args.results_dir) if losses.__len__() > 3 else None

		# lineplot(metrics['episodes'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
		# lineplot(metrics['episodes'][-len(metrics['reward_loss']):], metrics['reward_loss'], 'reward_loss', results_dir)
		# lineplot(metrics['episodes'][-len(metrics['kl_loss']):], metrics['kl_loss'], 'kl_loss', results_dir)
		# lineplot(metrics['episodes'][-len(metrics['actor_loss']):], metrics['actor_loss'], 'actor_loss', results_dir)
		# lineplot(metrics['episodes'][-len(metrics['value_loss']):], metrics['value_loss'], 'value_loss', results_dir)

	def save_model_data(self, episode):
		# writer.add_scalar("train_reward", metrics['train_rewards'][-1], metrics['steps'][-1])
		# writer.add_scalar("train/episode_reward", metrics['train_rewards'][-1], metrics['steps'][-1]*args.action_repeat)
		# writer.add_scalar("observation_loss", metrics['observation_loss'][0][-1], metrics['steps'][-1])
		# writer.add_scalar("reward_loss", metrics['reward_loss'][0][-1], metrics['steps'][-1])
		# writer.add_scalar("kl_loss", metrics['kl_loss'][0][-1], metrics['steps'][-1])
		# writer.add_scalar("actor_loss", metrics['actor_loss'][0][-1], metrics['steps'][-1])
		# writer.add_scalar("value_loss", metrics['value_loss'][0][-1], metrics['steps'][-1])
		# print("episodes: {}, total_steps: {}, train_reward: {} ".format(metrics['episodes'][-1], metrics['steps'][-1], metrics['train_rewards'][-1]))

		# Checkpoint models
		if episode % args.checkpoint_interval == 0:
			# torch.save({'transition_model': transition_model.state_dict(),
			#             'observation_model': observation_model.state_dict(),
			#             'reward_model': reward_model.state_dict(),
			#             'encoder': encoder.state_dict(),
			#             'actor_model': actor_model_g.state_dict(),
			#             'value_model': value_model_g.state_dict(),
			#             'model_optimizer': model_optimizer.state_dict(),
			#             'actor_optimizer': actor_optimizer_g.state_dict(),
			#             'value_optimizer': value_optimizer_g.state_dict()
			#             }, os.path.join(results_dir, 'models_%d.pth' % episode))
			if args.checkpoint_experience:
				torch.save(self.D, os.path.join(args.results_dir, 'experience.pth'))  # Warning: will fail with MemoryError with large memory sizes


if __name__ == "__main__":
	mp.set_start_method("spawn")
	# args.MultiGPU = False
	# os.environ['CUDA_VISIBLE_DEVICES'] = '3,2,1'
	torch.cuda.empty_cache()
	print(torch.cuda.is_available())
	print(torch.cuda.device_count())

	Plan().run()
