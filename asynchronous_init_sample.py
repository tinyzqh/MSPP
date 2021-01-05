from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher
import multiprocessing as mp
from parameter import args

class Worker_init_Sample(mp.Process):

  def __init__(self, child_conn, id):
    super(Worker_init_Sample, self).__init__()
    self.process_id = id
    self.child_conn = child_conn
  def run(self) -> None:
    sub_datas = self.child_conn.recv()
    env = Env(args.env, args.symbolic_env, args.seed, args.max_episode_length, args.action_repeat, args.bit_depth)
    observation, done, t = env.reset(), False, 0
    while not done:
      action = env.sample_random_action()
      next_observation, reward, done = env.step(action)
      sub_datas.append((observation, action, reward, done))
      observation = next_observation
      t += 1
      if t == 20:
        done = True
    sub_datas.append(t)
    self.child_conn.send(sub_datas)
    self.child_conn.close()
    env.close()

