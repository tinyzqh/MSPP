import torch
import time
from torch.nn import functional as F
from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher
from typing import Iterable
from torch.nn import Module
from parameter import args
import pandas as pd
import torch.multiprocessing as mp
from utils import lineplot, write_video, imagine_ahead, lambda_return, FreezeParameters, Save_Txt, ActivateParameters, get_modules
from models import bottle
from torch import nn, optim
from torch.distributions import Normal
from models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel, ValueModel, ActorModel

class Worker_actor(mp.Process):

    def __init__(self, actor_l, value_l, transition_model, encoder, observation_model, reward_model, child_conn, results_dir, id):
        super(Worker_actor, self).__init__()
        self.process_id = id
        self.actor_l, self.value_l = actor_l, value_l
        self.encoder, self.reward_model, self.transition_model, self.observation_model = encoder, reward_model, transition_model, observation_model

        self.child_conn = child_conn

        # Get model_modules
        self.env_model_modules = get_modules([self.transition_model, self.encoder, self.observation_model, self.reward_model])
        self.value_model_l_modules = get_modules([self.value_l])

        self.actor_optimizer_l = optim.Adam(self.actor_l.parameters(), lr=0 if args.learning_rate_schedule != 0 else args.actor_learning_rate, eps=args.adam_epsilon)
        self.value_optimizer_g = optim.Adam(self.value_l.parameters(), lr=0 if args.learning_rate_schedule != 0 else args.value_learning_rate, eps=args.adam_epsilon)

        self.metrics = {'episodes': [], 'actor_loss': [], 'value_loss': []}
        self.metrics['episodes'].extend([i for i in range(1, args.seed_episodes + 1)])
        self.losses = []
        self.count = 0
        self.results_dir = results_dir

        # print(self.prev_belief)
    def _flatten(self, x):
        return x.view([-1] + list(x.size()[2:]))
    def run(self) -> None:
        # print("children process {} waiting to get data".format(self.process_id))
        # Run = self.child_conn.recv()
        # print("children process {} Geted data form parent".format(self.process_id))
        while self.child_conn.recv() == 1:
            # print("Start Multi actor-critic Processing, The Process ID is {} -------------------------------".format(self.process_id))
            with FreezeParameters(self.env_model_modules):
                actor_states = torch.load("/home/hzq/Master's_thesis/tensor_data/actor_states.pt")
                actor_beliefs = torch.load("/home/hzq/Master's_thesis/tensor_data/actor_beliefs.pt")

                prev_state = self._flatten(actor_states.cuda()) if torch.cuda.is_available() and not args.disable_cuda else self._flatten(actor_states.cpu())
                prev_belief = self._flatten(actor_beliefs.cuda()) if torch.cuda.is_available() and not args.disable_cuda else self._flatten(actor_beliefs.cpu())

                # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
                T = args.planning_horizon
                beliefs, prior_states, prior_means, prior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
                beliefs[0], prior_states[0] = prev_belief, prev_state

                action_repate = []
                # Loop over time sequence
                for t in range(T - 1):
                    _state = prior_states[t]

                    ## add action repeat algorithms
                    if t % self.process_id == 0:
                        actions = self.actor_l.get_action(beliefs[t].detach(), _state.detach())
                        for i in range(self.process_id):
                            action_repate.append(actions)
                    if action_repate.__len__() != 0:
                        actions = action_repate.pop()

                    # Compute belief (deterministic hidden state)
                    if torch.cuda.device_count() > 1 and args.MultiGPU:
                        hidden = self.transition_model.module.act_fn(self.transition_model.module.fc_embed_state_action(torch.cat([_state, actions], dim=1)))
                        beliefs[t + 1] = self.transition_model.module.rnn(hidden, beliefs[t])
                        # Compute state prior by applying transition dynamics
                        hidden = self.transition_model.module.act_fn(self.transition_model.module.fc_embed_belief_prior(beliefs[t + 1]))
                        prior_means[t + 1], _prior_std_dev = torch.chunk(self.transition_model.module.fc_state_prior(hidden), 2, dim=1)
                        prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.transition_model.module.min_std_dev
                    else:
                        hidden = self.transition_model.act_fn(self.transition_model.fc_embed_state_action(torch.cat([_state, actions], dim=1)))
                        beliefs[t + 1] = self.transition_model.rnn(hidden, beliefs[t])
                        # Compute state prior by applying transition dynamics
                        hidden = self.transition_model.act_fn(self.transition_model.fc_embed_belief_prior(beliefs[t + 1]))
                        prior_means[t + 1], _prior_std_dev = torch.chunk(self.transition_model.fc_state_prior(hidden), 2, dim=1)
                        prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.transition_model.min_std_dev
                    prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
                    # Return new hidden states

                imged_beliefs, imged_prior_states, imged_prior_means, imged_prior_std_devs = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0),
                                 torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]

            # Update model parameters
            with FreezeParameters(self.env_model_modules + self.value_model_l_modules):
                imged_reward = bottle(self.reward_model, (imged_beliefs, imged_prior_states))
                value_pred = bottle(self.reward_model, (imged_beliefs, imged_prior_states))

            returns = lambda_return(imged_reward, value_pred, bootstrap=value_pred[-1], discount=args.discount, lambda_=args.disclam)
            actor_loss = -torch.mean(returns)

            # calculate local gradients and push local parameters to global
            self.actor_optimizer_l.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_l.parameters(), args.grad_clip_norm, norm_type=2)
            # for la, ga in zip(self.actor_l.parameters(), self.actor_g.parameters()):
            #     ga._grad = la.grad
            self.actor_optimizer_l.step()

            # push global parameters
            # self.actor_l.load_state_dict(self.actor_g.state_dict())

            # Dreamer implementation: value loss calculation and optimization
            with torch.no_grad():
                value_beliefs = imged_beliefs.detach()
                value_prior_states = imged_prior_states.detach()
                target_return = returns.detach()

            value_dist = Normal(bottle(self.value_l, (value_beliefs, value_prior_states)), 1)  # detach the input tensor from the transition network.
            value_loss = -value_dist.log_prob(target_return).mean(dim=(0, 1))
            # Update model parameters
            self.value_l.zero_grad()
            value_loss.backward()
            nn.utils.clip_grad_norm_(self.value_l.parameters(), args.grad_clip_norm, norm_type=2)
            self.value_optimizer_g.step()

            # save the loss data
            self.losses.append([actor_loss.item(), value_loss.item()])
            if self.count == args.collect_interval-1:
                losses = tuple(zip(*self.losses))
                self.metrics['actor_loss'].append(losses[0])
                self.metrics['value_loss'].append(losses[1])
                Save_Txt(self.metrics['episodes'][-1], self.metrics['actor_loss'][-1], 'actor_loss', self.results_dir)
                Save_Txt(self.metrics['episodes'][-1], self.metrics['value_loss'][-1], 'value_loss', self.results_dir)
                self.count = 0

            # print("End Multi actor-critic Processing, The Process ID is {} -------------------------------".format(self.process_id))
            self.count += 1
            self.metrics['episodes'].append(self.metrics['episodes'][-1] +1)
            self.child_conn.send(1)

