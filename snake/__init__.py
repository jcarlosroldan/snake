from snake.agent import Agent
from snake.environment import SnakeEnv
from snake.train import play_episode, ReplayBuffer, train
from snake.utils import checkpoint, log, plot, record_video

def run(device=None):
	agent = Agent(board_size=15, hidden_sizes=(675, 225, 64, 32, 16, 8), device=device)
	env = SnakeEnv(board_size=15, empty_size=5, max_turns=100, device=device)
	buffer = ReplayBuffer(board_size=15, length=10**6, device=device)
	losses, avg_rewards = [], []
	log('train', f'Started new training with an agent with {sum(p.numel() for p in agent.parameters()):,} parameters')
	for empty_size in range(5, 16):
		log('train', f'Training with {empty_size=}')
		env.empty_size = empty_size
		env.max_turns = 4 * empty_size**2
		log('train', 'Training with linearly decreasing epsilon')
		new_losses, new_avg_rewards = train(env, agent, buffer, episodes=100_000, batch_size=1000, lr=2e-5, max_eps=.3, gamma=.9)
		losses.append(new_losses)
		avg_rewards.append(new_avg_rewards)
		log('train', 'Training with epsilon=0')
		new_losses, new_avg_rewards = train(env, agent, buffer, episodes=10_000, batch_size=1000, lr=2e-5, max_eps=0, gamma=.9)
		losses.append(new_losses)
		avg_rewards.append(new_avg_rewards)
		checkpoint(agent, empty_size)
		plot(losses, 'Loss')
		plot(avg_rewards, 'Average reward')
		env.max_turns = 2000
		boards = play_episode(env, agent, eps=0, gamma=0)[0]
		record_video(boards, empty_size)
	return losses, avg_rewards