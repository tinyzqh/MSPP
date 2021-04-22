from dreamer.algorithms.plan_to_plan import Algorithms


class Algorithms_actor(Algorithms):
	def __init__(self, action_size, transition_model, encoder, reward_model, observation_model):
		super().__init__(action_size, transition_model, encoder, reward_model, observation_model)

	def get_action(self, belief, posterior_state, explore=False):
		action = self.actor_pool[0].get_action(belief, posterior_state, det=not (explore))
		return action
