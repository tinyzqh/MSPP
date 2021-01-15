import argparse
from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher
from torch.nn import functional as F
# Hyperparameters
parser = argparse.ArgumentParser(description='PlaNet or Dreamer')
# parser.add_argument('--algo', type=str, default='dreamer', help='planet or dreamer')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')

parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length')
parser.add_argument('--experience-size', type=int, default=1000000, metavar='D', help='Experience replay size')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
parser.add_argument('--cnn-activation-function', type=str, default='relu', choices=dir(F), help='Model activation function for a convolution layer')
parser.add_argument('--dense-activation-function', type=str, default='elu', choices=dir(F), help='Model activation function a dense layer')
parser.add_argument('--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
parser.add_argument('--hidden-size', type=int, default=200, metavar='H', help='Hidden size')
parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
parser.add_argument('--action-repeat', type=int, default=2, metavar='R', help='Action repeat')
parser.add_argument('--action-noise', type=float, default=0.3, metavar='ε', help='Action noise')
parser.add_argument('--episodes', type=int, default=1000, metavar='E', help='Total number of episodes')
parser.add_argument('--seed-episodes', type=int, default=5, metavar='S', help='Seed episodes')
parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval')
parser.add_argument('--chunk-size', type=int, default=50, metavar='L', help='Chunk size')
parser.add_argument('--overshooting-distance', type=int, default=50, metavar='D', help='Latent overshooting distance/latent overshooting weight for t = 1')
parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
parser.add_argument('--bit-depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
parser.add_argument('--model_learning-rate', type=float, default=1e-3, metavar='α', help='Learning rate') # 6 10-4
parser.add_argument('--actor_learning-rate', type=float, default=8e-5, metavar='α', help='Learning rate')
parser.add_argument('--value_learning-rate', type=float, default=8e-5, metavar='α', help='Learning rate')
parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS', help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)')
parser.add_argument('--adam-epsilon', type=float, default=1e-7, metavar='ε', help='Adam optimizer epsilon value')
# Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
parser.add_argument('--grad-clip-norm', type=float, default=100.0, metavar='C', help='Gradient clipping norm')
parser.add_argument('--planning-horizon', type=int, default=15, metavar='H', help='Planning horizon distance')
parser.add_argument('--discount', type=float, default=0.99, metavar='H', help='Planning horizon distance')
parser.add_argument('--disclam', type=float, default=0.95, metavar='H', help='discount rate to compute return')
parser.add_argument('--optimisation-iters', type=int, default=10, metavar='I', help='Planning optimisation iterations')
parser.add_argument('--candidates', type=int, default=1000, metavar='J', help='Candidate samples per iteration')
parser.add_argument('--top-candidates', type=int, default=100, metavar='K', help='Number of top candidates to fit')
parser.add_argument('--test', action='store_true', help='Test only')
parser.add_argument('--test-interval', type=int, default=50, metavar='I', help='Test interval (episodes)')
parser.add_argument('--test-episodes', type=int, default=9, metavar='E', help='Number of test episodes')
parser.add_argument('--checkpoint-interval', type=int, default=50, metavar='I', help='Checkpoint interval (episodes)')
parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay')
parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
parser.add_argument('--experience-replay', type=str, default='', metavar='ER', help='Load experience replay')
parser.add_argument('--render', action='store_true', help='Render environment')
parser.add_argument('--worldmodel-LogProbLoss', action='store_true', help='use LogProb loss for observation_model and reward_model training')

parser.add_argument('--overshooting-kl-beta', type=float, default=1, metavar='β>1',help='Latent overshooting KL weight for t > 1 (0 to disable)')
parser.add_argument('--overshooting-reward-scale', type=float, default=0, metavar='R>1', help='Latent overshooting reward prediction weight for t > 1 (0 to disable)')
parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg', help='Global KL weight (0 to disable)')

parser.add_argument('--MultiGPU', type=bool, default=True, help='if use multi gpu')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
# parser.add_argument('--MergeAgent', type=bool, default=True, help='if use multi gpu')

# ['cartpole-balance', 'cartpole-swingup', 'reacher-easy', 'finger-spin', 'cheetah-run', 'ball_in_cup-catch', 'walker-walk','reacher-hard', 'walker-run', 'humanoid-stand', 'humanoid-walk', 'fish-swim', 'acrobot-swingup']
parser.add_argument('--seed', type=int, default=5, metavar='S', help='Random seed')
parser.add_argument('--env', type=str, default='cartpole-balance', choices=GYM_ENVS + CONTROL_SUITE_ENVS, help='Gym/Control Suite environment')
parser.add_argument('--algo', type=str, default='p2p', help='planet, dreamer, p2p, actor_pool_1, dreamer_two_repeat')

parser.add_argument('--pool_len', type=int, default=2, help='number of sub actor in actor_pool')
parser.add_argument('--batch-size', type=int, default=36, metavar='B', help='Batch size')
parser.add_argument('--results_dir', type=str, default='some_error_happened', help="if the value == default, there may be some error")
parser.add_argument('--action-scale', type=int, default=-1, metavar='TS', help='Action scale; Only used in dreamer, IF Not in dreamer, set -1.')
parser.add_argument('--sub-traintime', type=int, default=5, metavar='TT', help='train time of sub actor.')
args = parser.parse_args()