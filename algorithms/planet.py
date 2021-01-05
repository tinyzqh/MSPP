import torch.nn as nn
from parameter import args
from torch import jit
import torch
from parameter import args

# Model-predictive control planner with cross-entropy method and learned transition model
class MPCPlanner(nn.Module):
  __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates']

  def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model, reward_model):
    super().__init__()
    self.transition_model, self.reward_model = transition_model, reward_model
    self.action_size = action_size
    self.planning_horizon = planning_horizon
    self.optimisation_iters = optimisation_iters
    self.candidates, self.top_candidates = candidates, top_candidates

  def upper_transition_model(self, prev_state, actions, prev_belief, obs=None, nonterminals=None):
    actions = torch.transpose(actions, 0, 1) if args.MultiGPU else actions
    nonterminals = torch.transpose(nonterminals, 0, 1).cuda() if args.MultiGPU and nonterminals is not None else nonterminals
    obs = torch.transpose(obs, 0, 1).cuda() if args.MultiGPU and obs is not None else obs
    temp_val = self.transition_model(prev_state.cuda(), actions.cuda(), prev_belief.cuda(), obs, nonterminals)
    return list(map(lambda x: x.view(-1, prev_state.shape[0], x.shape[2]), [x for x in temp_val]))

  # @jit.script_method
  def forward(self, belief, state):
    B, H, Z = belief.size(0), belief.size(1), state.size(1)
    belief, state = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H), state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)
    # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
    action_mean, action_std_dev = torch.zeros(self.planning_horizon, B, 1, self.action_size, device=belief.device), torch.ones(self.planning_horizon, B, 1, self.action_size, device=belief.device)
    for _ in range(self.optimisation_iters):
      # print("optimization_iters",_)
      # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
      actions = (action_mean + action_std_dev * torch.randn(self.planning_horizon, B, self.candidates, self.action_size, device=action_mean.device)).view(self.planning_horizon, B * self.candidates, self.action_size)  # Sample actions (time x (batch x candidates) x actions)
      # Sample next states

      beliefs, states, _, _  = self.upper_transition_model(state, actions, belief)

      # if args.MultiGPU:
      #   actions_trans = torch.transpose(actions, 0, 1).cuda()
      #   beliefs, states, _, _ = self.transition_model(state, actions_trans, belief)
      #   beliefs, states = list(map(lambda x: x.view(-1, self.candidates, x.shape[2]), [beliefs, states]))
      #
      # else:
      #   beliefs, states, _, _ = self.transition_model(state, actions, belief)

      # beliefs, states, _, _ = self.transition_model(state, actions, belief)# [12, 1000, 200] [12, 1000, 30] : 12 horizon steps; 1000 candidates
      # Calculate expected returns (technically sum of rewards over planning horizon)
      returns = self.reward_model(beliefs.view(-1, H), states.view(-1, Z)).view(self.planning_horizon, -1).sum(dim=0) #output from r-model[12000]->view[12, 1000]->sum[1000]
      # Re-fit belief to the K best action sequencessetting -> Repositories
      _, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
      topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)  # Fix indices for unrolled actions
      best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates, self.action_size)
      # Update belief with new means and standard deviations
      action_mean, action_std_dev = best_actions.mean(dim=2, keepdim=True), best_actions.std(dim=2, unbiased=False, keepdim=True)
    # Return first action mean µ_t
    return action_mean[0].squeeze(dim=1)


class Algorithms(object):
  def __init__(self, action_size, transition_model, reward_model):
    self.planner = MPCPlanner(action_size, args.planning_horizon, args.optimisation_iters, args.candidates, args.top_candidates, transition_model, reward_model)

  def get_action(self, belief, posterior_state, explore=False):
    action = self.planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
    # if args.algo == "dreamer":
    #   action = self.planner.get_action(belief, posterior_state, det=not (explore))
    # elif args.algo == "p2p":
    #   merge_action_list = []
    #   for actor_l in self.actor_pool:
    #     actions_l_mean, actions_l_std = actor_l.get_action_mean_std(belief, posterior_state)
    #     merge_action_list.append(actions_l_mean)
    #     merge_action_list.append(actions_l_std)
    #   merge_actions = torch.cat(merge_action_list, dim=1)
    #   action = self.planner.get_merge_action(merge_actions, belief, posterior_state)
    # elif args.algo == "planet":
    #   action = self.planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
    # elif args.algo == "actor_pool_1":
    #   action = self.planner.get_action(belief, posterior_state, det=not (explore))
    # else:
    #   action = self.planner.get_action(belief, posterior_state, det=not (explore))
    return action

  def train_algorithm(self, actor_states, actor_beliefs):
    "the planet no train step"
    return None

  def save_loss_data(self, metrics_episodes):
    "the loss no loss data"
    return None

  def train_to_eval(self):
    "the planet no train_to_eval step"
    return None

  def eval_to_train(self):
    "the planet no eval_to_train step"
    return None