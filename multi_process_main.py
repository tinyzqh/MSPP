import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from dreamer.env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher
from dreamer.memory import ExperienceReplay
from dreamer.models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel, ValueModel, ActorModel
from dreamer.planner import MPCPlanner
from dreamer.utils import lineplot, write_video, imagine_ahead, lambda_return, FreezeParameters, Save_Txt, ActivateParameters
from tensorboardX import SummaryWriter

from dreamer.parameter import args

args.overshooting_distance = min(args.chunk_size, args.overshooting_distance)  # Overshooting distance cannot be greater than chunk size
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

print("torch.cuda.device_count() {}".format(torch.cuda.device_count()))

# Setup

results_dir = os.path.join('results', '{}_{}_{}'.format(args.env, args.seed, args.algo))
os.makedirs(results_dir, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
  print("using CUDA")
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(args.seed)
else:
  print("using CPU")
  args.device = torch.device('cpu')
metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [],
           'observation_loss': [], 'reward_loss': [], 'kl_loss': [], 'actor_loss': [], 'value_loss': []}

summary_name = results_dir + "/{}_{}_log"
writer = SummaryWriter(summary_name.format(args.env, args.id))

def job(temp, i):
  print("Start the SubProcess {}".format(i))
  # temp = []
  env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth)

  observation, done, t = env.reset(), False, 0
  while not done:
    action = env.sample_random_action()
    next_observation, reward, done = env.step(action)
    temp.append((observation, action, reward, done))
    observation = next_observation
    t += 1
    if t == 20:
      done = True
    # done=True
  temp.append(t)
  env.close()
  print("End the SubProcess {}".format(i))

if __name__ == "__main__":
  import multiprocessing as mp
  import time
  # Initialise training environment and experience replay memory
  env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth)

  if args.experience_replay is not '' and os.path.exists(args.experience_replay):
    D = torch.load(args.experience_replay)
    metrics['steps'], metrics['episodes'] = [D.steps] * D.episodes, list(range(1, D.episodes + 1))
  elif not args.test:
    D = ExperienceReplay(args.experience_size, args.symbolic_env, env.observation_size, env.action_size, args.bit_depth, args.device)
    # Initialise dataset D with S random seed episodes

    # Initialise dataset D with S random seed episodes
    print("Start Multi Processing -------------------------------")
    start_time = time.time()
    manager = mp.Manager()
    temp_lists = [manager.list() for i in range(args.seed_episodes)]
    process_list = []
    for i in range(1, args.seed_episodes + 1):
      p = mp.Process(target=job, args=(temp_lists[i - 1], i,))
      p.start()
      process_list.append(p)
    for s, p in enumerate(process_list):
      p.join()
    for s, temp_list in enumerate(temp_lists):
      for data in list(temp_list):
        if type(data) == tuple:
          assert len(data) == 4
          # print("data {}".format(data))
          D.append(data[0], data[1], data[2], data[3])
        else:
          assert type(data) == int
          t = data
          metrics['steps'].append(t * args.action_repeat + (0 if len(metrics['steps']) == 0 else metrics['steps'][-1]))
          metrics['episodes'].append(s + 1)
    end_time = time.time()
    print("the process times {} s".format(end_time - start_time))
    print("End Multi Processing -------------------------------")

  # Initialise model parameters randomly
  transition_model = TransitionModel(args.belief_size, args.state_size, env.action_size, args.hidden_size, args.embedding_size, args.dense_activation_function).to(device=args.device)
  observation_model = ObservationModel(args.symbolic_env, env.observation_size, args.belief_size, args.state_size, args.embedding_size, args.cnn_activation_function).to(device=args.device)
  reward_model = RewardModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(device=args.device)
  encoder = Encoder(args.symbolic_env, env.observation_size, args.embedding_size, args.cnn_activation_function).to(device=args.device)
  actor_model = ActorModel(args.belief_size, args.state_size, args.hidden_size, env.action_size, args.dense_activation_function).to(device=args.device)
  value_model = ValueModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(device=args.device)

  if torch.cuda.device_count() > 1 and args.MultiGPU:
    print("We Have {} GPUS".format(torch.cuda.device_count()))
    transition_model = nn.DataParallel(transition_model.cuda())
    observation_model = nn.DataParallel(observation_model.cuda())
    reward_model = nn.DataParallel(reward_model.cuda())
    # encoder = nn.DataParallel(encoder.cuda(), device_ids=[0, 1, 2])
    # actor_model = nn.DataParallel(actor_model.cuda())
    # value_model = nn.DataParallel(value_model.cuda())

  param_list = list(transition_model.parameters()) + list(observation_model.parameters()) + list(reward_model.parameters()) + list(encoder.parameters())
  value_actor_param_list = list(value_model.parameters()) + list(actor_model.parameters())
  params_list = param_list + value_actor_param_list
  model_optimizer = optim.Adam(param_list, lr=0 if args.learning_rate_schedule != 0 else args.model_learning_rate, eps=args.adam_epsilon)
  actor_optimizer = optim.Adam(actor_model.parameters(), lr=0 if args.learning_rate_schedule != 0 else args.actor_learning_rate, eps=args.adam_epsilon)
  value_optimizer = optim.Adam(value_model.parameters(), lr=0 if args.learning_rate_schedule != 0 else args.value_learning_rate, eps=args.adam_epsilon)

  if args.algo=="dreamer":
    print("DREAMER")
    planner = actor_model
  else:
    planner = MPCPlanner(env.action_size, args.planning_horizon, args.optimisation_iters, args.candidates, args.top_candidates, transition_model, reward_model)
  global_prior = Normal(torch.zeros(args.batch_size, args.state_size, device=args.device), torch.ones(args.batch_size, args.state_size, device=args.device))  # Global prior N(0, I)
  free_nats = torch.full((1, ), args.free_nats, device=args.device)  # Allowed deviation in KL divergence


  def update_belief_and_act_multi_gpu(args, env, planner, transition_model, encoder, belief, posterior_state, action, observation, explore=False):
    # Infer belief over current state q(s_t|o≤t,a<t) from the history
    # print("action size: ",action.size()) torch.Size([1, 6])
    if args.MultiGPU:
      # transpose batch_size first for DataParallel
      actions_trans = torch.transpose(action.unsqueeze(dim=0), 0, 1).cuda()
      obs_trans = torch.transpose(encoder(observation).unsqueeze(dim=0), 0, 1).cuda()

      # Infer belief over current state q(s_t|o≤t,a<t) from the history
      belief, _, _, _, posterior_state, _, _ = transition_model(posterior_state, actions_trans, belief, obs_trans)  # Action and observation need extra time dimension

      if hasattr(env, "envs"):
        belief, posterior_state = list(map(lambda x: x.view(-1, args.test_episodes, x.shape[2]), [x for x in [belief, posterior_state]]))

    else:
      # Infer belief over current state q(s_t|o≤t,a<t) from the history
      belief, _, _, _, posterior_state, _, _ = transition_model(posterior_state, action.unsqueeze(dim=0), belief,
                                                                encoder(observation).unsqueeze(dim=0))  # Action and observation need extra time dimension
    # belief, _, _, _, posterior_state, _, _ = transition_model(posterior_state, action.unsqueeze(dim=0), belief, encoder(observation).unsqueeze(dim=0))  # Action and observation need extra time dimension
    belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)  # Remove time dimension from belief/state
    if args.algo=="dreamer":
      action = planner.get_action(belief, posterior_state, det=not(explore))
    else:
      action = planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
    if explore:
      action = torch.clamp(Normal(action, args.action_noise).rsample(), -1, 1) # Add gaussian exploration noise on top of the sampled action
      # action = action + args.action_noise * torch.randn_like(action)  # Add exploration noise ε ~ p(ε) to the action
    next_observation, reward, done = env.step(action.cpu() if isinstance(env, EnvBatcher) else action[0].cpu())  # Perform environment step (action repeats handled internally)
    return belief, posterior_state, action, next_observation, reward, done

  # Testing only
  if args.test:
    # Set models to eval mode
    transition_model.eval()
    reward_model.eval()
    encoder.eval()
    with torch.no_grad():
      total_reward = 0
      for _ in tqdm(range(args.test_episodes)):
        observation = env.reset()
        belief, posterior_state, action = torch.zeros(1, args.belief_size, device=args.device), torch.zeros(1, args.state_size, device=args.device), torch.zeros(1, env.action_size, device=args.device)
        pbar = tqdm(range(args.max_episode_length // args.action_repeat))
        for t in pbar:
          belief, posterior_state, action, observation, reward, done = update_belief_and_act_multi_gpu(args, env, planner, transition_model, encoder, belief, posterior_state, action, observation.to(device=args.device))
          total_reward += reward
          if args.render:
            env.render()
          if done:
            pbar.close()
            break
    print('Average Reward:', total_reward / args.test_episodes)
    env.close()
    quit()

  # Training (and testing)
  for episode in tqdm(range(metrics['episodes'][-1] + 1, args.episodes + 1), total=args.episodes, initial=metrics['episodes'][-1] + 1):

    # Model fitting
    losses = []
    if torch.cuda.device_count() > 1 and args.MultiGPU:
      model_modules = list(transition_model.module.modules()) + list(encoder.module.modules()) + list(
        observation_model.module.modules()) + list(reward_model.module.modules())
    else:
      model_modules = transition_model.modules + encoder.modules + observation_model.modules + reward_model.modules

    print("training loop")
    # args.collect_interval = 1
    for s in tqdm(range(args.collect_interval)):
      # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
      observations, actions, rewards, nonterminals = D.sample(args.batch_size, args.chunk_size) # Transitions start at time t = 0
      # Create initial belief and state for time t = 0
      init_belief, init_state = torch.zeros(args.batch_size, args.belief_size, device=args.device), torch.zeros(args.batch_size, args.state_size, device=args.device)
      # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
      obs = bottle(encoder, (observations[1:],))
      if args.MultiGPU:
        actions_trans = torch.transpose(actions[:-1], 0, 1)
        obs_trans = torch.transpose(obs, 0, 1)
        nonterminals_trans = torch.transpose(nonterminals[:-1], 0, 1)

        temp_val = transition_model(
          prev_state=init_state.cuda(),
          actions=actions_trans.cuda(),
          prev_belief=init_belief.cuda(),
          observations=obs_trans.cuda(),
          nonterminals=nonterminals_trans.cuda())

        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = list(
          map(lambda x: x.view(-1, args.batch_size, x.shape[2]), [x for x in temp_val]))

      else:
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = transition_model(
          init_state, actions[:-1], init_belief, obs, nonterminals[:-1])
      # beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = transition_model(init_state, actions[:-1], init_belief, bottle(encoder, (observations[1:], )), nonterminals[:-1])

      # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
      if args.worldmodel_LogProbLoss:
        observation_dist = Normal(bottle(observation_model, (beliefs, posterior_states)), 1)
        observation_loss = -observation_dist.log_prob(observations[1:]).sum(dim=2 if args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))
      else:
        observation_loss = F.mse_loss(bottle(observation_model, (beliefs, posterior_states)), observations[1:], reduction='none').sum(dim=2 if args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))
      if args.worldmodel_LogProbLoss:
        reward_dist = Normal(bottle(reward_model, (beliefs, posterior_states)),1)
        reward_loss = -reward_dist.log_prob(rewards[:-1]).mean(dim=(0, 1))
      else:
        reward_loss = F.mse_loss(bottle(reward_model, (beliefs, posterior_states)), rewards[:-1], reduction='none').mean(dim=(0,1))
      # transition loss
      div = kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2)
      kl_loss = torch.max(div, free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
      if args.global_kl_beta != 0:
        kl_loss += args.global_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs), global_prior).sum(dim=2).mean(dim=(0, 1))
      # Calculate latent overshooting objective for t > 0
      if args.overshooting_kl_beta != 0:
        overshooting_vars = []  # Collect variables for overshooting to process in batch
        for t in range(1, args.chunk_size - 1):
          d = min(t + args.overshooting_distance, args.chunk_size - 1)  # Overshooting distance
          t_, d_ = t - 1, d - 1  # Use t_ and d_ to deal with different time indexing for latent states
          seq_pad = (0, 0, 0, 0, 0, t - d + args.overshooting_distance)  # Calculate sequence padding so overshooting terms can be calculated in one batch
          # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
          overshooting_vars.append((F.pad(actions[t:d], seq_pad), F.pad(nonterminals[t:d], seq_pad), F.pad(rewards[t:d], seq_pad[2:]), beliefs[t_], prior_states[t_], F.pad(posterior_means[t_ + 1:d_ + 1].detach(), seq_pad), F.pad(posterior_std_devs[t_ + 1:d_ + 1].detach(), seq_pad, value=1), F.pad(torch.ones(d - t, args.batch_size, args.state_size, device=args.device), seq_pad)))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
        overshooting_vars = tuple(zip(*overshooting_vars))
        # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
        beliefs, prior_states, prior_means, prior_std_devs = transition_model(torch.cat(overshooting_vars[4], dim=0), torch.cat(overshooting_vars[0], dim=1), torch.cat(overshooting_vars[3], dim=0), None, torch.cat(overshooting_vars[1], dim=1))
        seq_mask = torch.cat(overshooting_vars[7], dim=1)
        # Calculate overshooting KL loss with sequence mask
        kl_loss += (1 / args.overshooting_distance) * args.overshooting_kl_beta * torch.max((kl_divergence(Normal(torch.cat(overshooting_vars[5], dim=1), torch.cat(overshooting_vars[6], dim=1)), Normal(prior_means, prior_std_devs)) * seq_mask).sum(dim=2), free_nats).mean(dim=(0, 1)) * (args.chunk_size - 1)  # Update KL loss (compensating for extra average over each overshooting/open loop sequence)
        # Calculate overshooting reward prediction loss with sequence mask
        if args.overshooting_reward_scale != 0:
          reward_loss += (1 / args.overshooting_distance) * args.overshooting_reward_scale * F.mse_loss(bottle(reward_model, (beliefs, prior_states)) * seq_mask[:, :, 0], torch.cat(overshooting_vars[2], dim=1), reduction='none').mean(dim=(0, 1)) * (args.chunk_size - 1)  # Update reward loss (compensating for extra average over each overshooting/open loop sequence)
      # Apply linearly ramping learning rate schedule
      if args.learning_rate_schedule != 0:
        for group in model_optimizer.param_groups:
          group['lr'] = min(group['lr'] + args.model_learning_rate / args.model_learning_rate_schedule, args.model_learning_rate)
      model_loss = observation_loss + reward_loss + kl_loss
      # Update model parameters
      model_optimizer.zero_grad()
      model_loss.backward()
      nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
      model_optimizer.step()

      #Dreamer implementation: actor loss calculation and optimization
      with torch.no_grad():
        actor_states = posterior_states.detach()
        actor_beliefs = beliefs.detach()
      with FreezeParameters(model_modules):
        imagination_traj = imagine_ahead(actor_states, actor_beliefs, actor_model, transition_model, args.planning_horizon)
      imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs = imagination_traj
      with FreezeParameters(model_modules + list(value_model.modules()) if args.MultiGPU else value_model.modules):
        imged_reward = bottle(reward_model, (imged_beliefs, imged_prior_states))
        value_pred = bottle(value_model, (imged_beliefs, imged_prior_states))
      returns = lambda_return(imged_reward, value_pred, bootstrap=value_pred[-1], discount=args.discount, lambda_=args.disclam)
      actor_loss = -torch.mean(returns)
      # Update model parameters
      actor_optimizer.zero_grad()
      actor_loss.backward()
      nn.utils.clip_grad_norm_(actor_model.parameters(), args.grad_clip_norm, norm_type=2)
      actor_optimizer.step()

      #Dreamer implementation: value loss calculation and optimization
      with torch.no_grad():
        value_beliefs = imged_beliefs.detach()
        value_prior_states = imged_prior_states.detach()
        target_return = returns.detach()
      value_dist = Normal(bottle(value_model, (value_beliefs, value_prior_states)),1) # detach the input tensor from the transition network.
      value_loss = -value_dist.log_prob(target_return).mean(dim=(0, 1))
      # Update model parameters
      value_optimizer.zero_grad()
      value_loss.backward()
      nn.utils.clip_grad_norm_(value_model.parameters(), args.grad_clip_norm, norm_type=2)
      value_optimizer.step()

      # # Store (0) observation loss (1) reward loss (2) KL loss (3) actor loss (4) value loss
      losses.append([observation_loss.item(), reward_loss.item(), kl_loss.item(), actor_loss.item(), value_loss.item()])

    # Update and plot loss metrics
    losses = tuple(zip(*losses))
    metrics['observation_loss'].append(losses[0])
    metrics['reward_loss'].append(losses[1])
    metrics['kl_loss'].append(losses[2])
    metrics['actor_loss'].append(losses[3])
    metrics['value_loss'].append(losses[4])

    Save_Txt(metrics['episodes'][-1], metrics['observation_loss'][-1], 'observation_loss',results_dir)
    Save_Txt(metrics['episodes'][-1], metrics['reward_loss'][-1], 'reward_loss', results_dir)
    Save_Txt(metrics['episodes'][-1], metrics['kl_loss'][-1], 'kl_loss', results_dir)
    Save_Txt(metrics['episodes'][-1], metrics['actor_loss'][-1], 'actor_loss', results_dir)
    Save_Txt(metrics['episodes'][-1], metrics['value_loss'][-1], 'value_loss', results_dir)

    # lineplot(metrics['episodes'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
    # lineplot(metrics['episodes'][-len(metrics['reward_loss']):], metrics['reward_loss'], 'reward_loss', results_dir)
    # lineplot(metrics['episodes'][-len(metrics['kl_loss']):], metrics['kl_loss'], 'kl_loss', results_dir)
    # lineplot(metrics['episodes'][-len(metrics['actor_loss']):], metrics['actor_loss'], 'actor_loss', results_dir)
    # lineplot(metrics['episodes'][-len(metrics['value_loss']):], metrics['value_loss'], 'value_loss', results_dir)

    # Data collection
    print("Data collection")
    with torch.no_grad():
      observation, total_reward = env.reset(), 0
      belief, posterior_state, action = torch.zeros(1, args.belief_size, device=args.device), torch.zeros(1, args.state_size, device=args.device), torch.zeros(1, env.action_size, device=args.device)
      pbar = tqdm(range(args.max_episode_length // args.action_repeat))
      for t in pbar:
        # print("step",t)
        belief, posterior_state, action, next_observation, reward, done = update_belief_and_act_multi_gpu(args, env, planner, transition_model, encoder, belief, posterior_state, action, observation.to(device=args.device), explore=True)
        D.append(observation, action.cpu(), reward, done)
        total_reward += reward
        observation = next_observation
        if args.render:
          env.render()
        if done:
          pbar.close()
          break

      # Update and plot train reward metrics
      metrics['steps'].append(t + metrics['steps'][-1])
      metrics['episodes'].append(episode)
      metrics['train_rewards'].append(total_reward)

      Save_Txt(metrics['episodes'][-1], metrics['train_rewards'][-1], 'train_rewards',results_dir)
      # lineplot(metrics['episodes'][-len(metrics['train_rewards']):], metrics['train_rewards'], 'train_rewards', results_dir)

    # Test model
    print("Test model")
    # args.test_interval = 1
    if episode % args.test_interval == 0:
      # Set models to eval mode
      transition_model.eval()
      observation_model.eval()
      reward_model.eval()
      encoder.eval()
      actor_model.eval()
      value_model.eval()
      # Initialise parallelised test environments
      test_envs = EnvBatcher(Env, (args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth), {}, args.test_episodes)

      with torch.no_grad():
        observation, total_rewards, video_frames = test_envs.reset(), np.zeros((args.test_episodes, )), []
        belief, posterior_state, action = torch.zeros(args.test_episodes, args.belief_size, device=args.device), torch.zeros(args.test_episodes, args.state_size, device=args.device), torch.zeros(args.test_episodes, env.action_size, device=args.device)
        pbar = tqdm(range(args.max_episode_length // args.action_repeat))
        for t in pbar:

          belief, posterior_state, action, next_observation, reward, done = update_belief_and_act_multi_gpu(args, test_envs, planner, transition_model,encoder, belief, posterior_state, action, observation.to(device=args.device))
          total_rewards += reward.numpy()
          if not args.symbolic_env:  # Collect real vs. predicted frames for video
            video_frames.append(make_grid(torch.cat([observation, observation_model(belief, posterior_state).cpu()], dim=3) + 0.5, nrow=5).numpy())  # Decentre
          observation = next_observation
          if done.sum().item() == args.test_episodes:
            pbar.close()
            break

      # Update and plot reward metrics (and write video if applicable) and save metrics
      metrics['test_episodes'].append(episode)
      metrics['test_rewards'].append(total_rewards.tolist())

      Save_Txt(metrics['test_episodes'][-1], metrics['test_rewards'][-1], 'test_rewards', results_dir)
      # Save_Txt(np.asarray(metrics['steps'])[np.asarray(metrics['test_episodes']) - 1], metrics['test_rewards'],'test_rewards_steps', results_dir, xaxis='step')

      # lineplot(metrics['test_episodes'], metrics['test_rewards'], 'test_rewards', results_dir)
      # lineplot(np.asarray(metrics['steps'])[np.asarray(metrics['test_episodes']) - 1], metrics['test_rewards'], 'test_rewards_steps', results_dir, xaxis='step')
      if not args.symbolic_env:
        episode_str = str(episode).zfill(len(str(args.episodes)))
        write_video(video_frames, 'test_episode_%s' % episode_str, results_dir)  # Lossy compression
        save_image(torch.as_tensor(video_frames[-1]), os.path.join(results_dir, 'test_episode_%s.png' % episode_str))
      torch.save(metrics, os.path.join(results_dir, 'metrics.pth'))

      # Set models to train mode
      transition_model.train()
      observation_model.train()
      reward_model.train()
      encoder.train()
      actor_model.train()
      value_model.train()
      # Close test environments
      test_envs.close()

    writer.add_scalar("train_reward", metrics['train_rewards'][-1], metrics['steps'][-1])
    writer.add_scalar("train/episode_reward", metrics['train_rewards'][-1], metrics['steps'][-1]*args.action_repeat)
    writer.add_scalar("observation_loss", metrics['observation_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("reward_loss", metrics['reward_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("kl_loss", metrics['kl_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("actor_loss", metrics['actor_loss'][0][-1], metrics['steps'][-1])
    writer.add_scalar("value_loss", metrics['value_loss'][0][-1], metrics['steps'][-1])
    print("episodes: {}, total_steps: {}, train_reward: {} ".format(metrics['episodes'][-1], metrics['steps'][-1], metrics['train_rewards'][-1]))

    # Checkpoint models
    # args.checkpoint_interval = 1
    if episode % args.checkpoint_interval == 0:
      torch.save({'transition_model': transition_model.state_dict(),
                  'observation_model': observation_model.state_dict(),
                  'reward_model': reward_model.state_dict(),
                  'encoder': encoder.state_dict(),
                  'actor_model': actor_model.state_dict(),
                  'value_model': value_model.state_dict(),
                  'model_optimizer': model_optimizer.state_dict(),
                  'actor_optimizer': actor_optimizer.state_dict(),
                  'value_optimizer': value_optimizer.state_dict()
                  }, os.path.join(results_dir, 'models_%d.pth' % episode))
      if args.checkpoint_experience:
        torch.save(D, os.path.join(results_dir, 'experience.pth'))  # Warning: will fail with MemoryError with large memory sizes

  # Close training environment
  env.close()