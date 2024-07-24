from snake.utils import log
from time import time
import torch

def train(env, agent, buffer, episodes, batch_size, lr, max_eps, gamma):
	optimizer = torch.optim.AdamW(agent.parameters(), lr=lr)
	losses, avg_rewards = [], []
	start = time()
	for e in range(episodes):
		progress = (e + 1) / episodes
		# record experiences
		agent.eval()
		buffer.append(*play_episode(env, agent, eps=max_eps * (1 - progress), gamma=gamma))
		# update agent
		agent.train()
		boards, actions, rewards = buffer.sample(batch_size)
		predicted_rewards = agent(boards)[torch.arange(len(boards)), actions]
		loss = torch.nn.functional.mse_loss(predicted_rewards, rewards)
		optimizer.zero_grad()
		loss.backward()
		# stats
		losses.append(loss.item())
		avg_rewards.append(rewards.mean().item())
		optimizer.step()
		if e % (episodes // 10) == 0 and e > 0 or e == 50 or e == episodes - 1:
			remaining = (time() - start) * (1 - progress) / progress
			log('train', f'[{100 * progress:5.2f}%, remaining {remaining // 60:02.0f}:{remaining % 60:02.0f}] loss {torch.tensor(losses[-1000:]).mean():.6f}, avg reward {torch.tensor(avg_rewards[-1000:]).mean():.6f}')
	return losses, avg_rewards

def play_episode(env, agent, eps=.3, gamma=.9, device=None):
	observations, actions, rewards = [], [], []
	observation = env.reset()
	done = False
	while not done:
		observations.append(observation.clone())
		if torch.rand(()) < eps:
			action = torch.randint(0, 4, ())
		else:
			action = agent(observation).argmax().item()
		actions.append(action)
		observation, reward, done = env.step(action)
		rewards.append(reward)
	for i in range(len(rewards) - 2, -1, -1):
		rewards[i] += gamma * rewards[i + 1]
	return torch.stack(observations), torch.tensor(actions, device=device), torch.tensor(rewards, dtype=torch.float, device=device)

class ReplayBuffer:
	def __init__(self, board_size, length, device=None):
		self.boards = torch.zeros((length, board_size, board_size), dtype=torch.int, device=device)
		self.actions = torch.zeros((length,), dtype=torch.int, device=device)
		self.rewards = torch.zeros((length,), dtype=torch.float, device=device)
		self.length = length
		self.next_ix = 0

	def append(self, boards, actions, rewards):
		batch_size = len(boards)
		last_ix = self.next_ix + batch_size
		if last_ix <= self.length:
			self.boards[self.next_ix:last_ix], self.actions[self.next_ix:last_ix], self.rewards[self.next_ix:last_ix] = boards, actions, rewards
			self.next_ix = last_ix % self.length
		else:
			fitting = self.length - self.next_ix
			self.boards[-fitting:], self.actions[-fitting:], self.rewards[-fitting:] = boards[:fitting], actions[:fitting], rewards[:fitting]
			self.next_ix = 0
			self.append(boards[fitting:], actions[fitting:], rewards[fitting:])

	def sample(self, batch_size):
		ix = torch.randint(0, self.length, (min(batch_size, self.next_ix),))
		return self.boards[ix], self.actions[ix], self.rewards[ix]