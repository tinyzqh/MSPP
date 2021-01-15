import torch
import os
from torch import nn, optim
from torch.nn import functional as F
from parameter import args
from models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel, ValueModel, ActorModel, MergeModel
from utils import lineplot, write_video, imagine_ahead, lambda_return, FreezeParameters, Save_Txt, ActivateParameters, get_modules
from torch.distributions import Normal

class Algorithms(object):

  def __init__(self,action_size, transition_model, encoder, reward_model, observation_model):
    self.encoder, self.reward_model, self.transition_model, self.observation_model = encoder, reward_model, transition_model, observation_model

    self.actor_model = ActorModel(args.belief_size, args.state_size, args.hidden_size, action_size, args.dense_activation_function).to(device=args.device)
    self.value_model = ValueModel(args.belief_size, args.state_size, args.hidden_size, args.dense_activation_function).to(device=args.device)
    self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=0 if args.learning_rate_schedule != 0 else args.actor_learning_rate, eps=args.adam_epsilon)
    self.value_optimizer = optim.Adam(self.value_model.parameters(), lr=0 if args.learning_rate_schedule != 0 else args.value_learning_rate, eps=args.adam_epsilon)

    # transition_model_modules = self.transition_model.model_modules if hasattr(self.transition_model, "model_modules") else list(self.transition_model.module.modules())
    # encoder_model_modules = self.encoder.model_modules if hasattr(self.encoder, "model_modules") else list(self.encoder.module.modules())
    # observation_model_modules = self.observation_model.model_modules if hasattr(self.observation_model, "model_modules") else list(self.observation_model.module.modules())
    # reward_model_modules = self.reward_model.model_modules if hasattr(self.reward_model, "model_modules") else list(self.reward_model.module.modules())
    # self.model_modules = transition_model_modules + encoder_model_modules + observation_model_modules + reward_model_modules

    self.env_model_modules = get_modules([self.transition_model, self.encoder, self.observation_model, self.reward_model])
    self.value_model_modules = get_modules([self.value_model])

    self.sub_actor_losses = []
    self.metrics = {'episodes': [], 'actor_loss': [], 'value_loss': []}

  def get_action(self, belief, posterior_state, explore=False):
    action = self.actor_model.get_action(belief, posterior_state, det=not (explore))
    return action

  def train_algorithm(self, actor_states, actor_beliefs):

    actor_states = torch.load(os.path.join(os.getcwd(), 'tensor_data/actor_states.pt'))
    actor_beliefs = torch.load(os.path.join(os.getcwd(), 'tensor_data/actor_beliefs.pt'))

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
  #
  # def imagine_ahead(self, prev_state, prev_belief, policy, transition_model, planning_horizon=12):
  #   '''
  #   imagine_ahead is the function to draw the imaginary tracjectory using the dynamics model, actor, critic.
  #   Input: current state (posterior), current belief (hidden), policy, transition_model  # torch.Size([50, 30]) torch.Size([50, 200])
  #   Output: generated trajectory of features includes beliefs, prior_states, prior_means, prior_std_devs
  #           torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
  #   '''
  #   flatten = lambda x: x.view([-1] + list(x.size()[2:]))
  #   prev_belief = flatten(prev_belief)
  #   prev_state = flatten(prev_state)
  #
  #   # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
  #   T = planning_horizon
  #   beliefs, prior_states, prior_means, prior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
  #   beliefs[0], prior_states[0] = prev_belief, prev_state
  #
  #   # Loop over time sequence
  #   for t in range(T - 1):
  #     _state = prior_states[t]
  #     # start_time = time.time()
  #     actions = policy.get_action(beliefs[t].detach(), _state.detach())
  #     # end_time = time.time()
  #     # print("the time is {}".format(end_time-start_time))
  #     # Compute belief (deterministic hidden state)
  #     if args.MultiGPU:
  #       hidden = transition_model.module.act_fn(
  #         transition_model.module.fc_embed_state_action(torch.cat([_state, actions], dim=1)))
  #       beliefs[t + 1] = transition_model.module.rnn(hidden, beliefs[t])
  #       # Compute state prior by applying transition dynamics
  #       hidden = transition_model.module.act_fn(transition_model.module.fc_embed_belief_prior(beliefs[t + 1]))
  #       prior_means[t + 1], _prior_std_dev = torch.chunk(transition_model.module.fc_state_prior(hidden), 2, dim=1)
  #       prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + transition_model.module.min_std_dev
  #     else:
  #       hidden = transition_model.act_fn(transition_model.fc_embed_state_action(torch.cat([_state, actions], dim=1)))
  #       beliefs[t + 1] = transition_model.rnn(hidden, beliefs[t])
  #       # Compute state prior by applying transition dynamics
  #       hidden = transition_model.act_fn(transition_model.fc_embed_belief_prior(beliefs[t + 1]))
  #       prior_means[t + 1], _prior_std_dev = torch.chunk(transition_model.fc_state_prior(hidden), 2, dim=1)
  #       prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + transition_model.min_std_dev
  #     prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
  #     # Return new hidden states
  #   # imagined_traj = [beliefs, prior_states, prior_means, prior_std_devs]
  #   imagined_traj = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0),
  #                    torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
  #   return imagined_traj

  def train_to_eval(self):
    self.actor_model.eval()
    self.value_model.eval()

  def eval_to_train(self):
    self.actor_model.train()
    self.value_model.train()

  # def get_action(self, belief, posterior_state, explore=False):
  #   action = self.planner.get_action(belief, posterior_state, det=not (explore))
  #   # if args.algo == "dreamer":
  #   #
  #   # elif args.algo == "p2p":
  #   #   merge_action_list = []
  #   #   for actor_l in self.actor_pool:
  #   #     actions_l_mean, actions_l_std = actor_l.get_action_mean_std(belief, posterior_state)
  #   #     merge_action_list.append(actions_l_mean)
  #   #     merge_action_list.append(actions_l_std)
  #   #   merge_actions = torch.cat(merge_action_list, dim=1)
  #   #   action = self.planner.get_merge_action(merge_actions, belief, posterior_state)
  #   # elif args.algo == "planet":
  #   #   action = self.planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
  #   # elif args.algo == "actor_pool_1":
  #   #   action = self.planner.get_action(belief, posterior_state, det=not (explore))
  #   # else:
  #   #   action = self.planner.get_action(belief, posterior_state, det=not (explore))
  #   return action