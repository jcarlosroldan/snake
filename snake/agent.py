import torch
import torch.nn as nn

class Agent(nn.Module):
	NUM_CLASSES = 8
	def __init__(self, board_size, hidden_sizes, device=None):
		super().__init__()
		self.register_buffer('diagonal', torch.eye(Agent.NUM_CLASSES, device=device))
		self.input = nn.Linear(board_size**2 * Agent.NUM_CLASSES, hidden_sizes[0])
		self.hidden = nn.Sequential()
		for h in range(len(hidden_sizes) - 1):
			self.hidden.extend([
				nn.Linear(hidden_sizes[h], hidden_sizes[h + 1]),
				nn.GELU(),
				nn.LayerNorm(hidden_sizes[h + 1])
			])
		self.output = nn.Linear(hidden_sizes[-1], 4, bias=False)

	def forward(self, x):
		x = self.diagonal[x.flatten(-2)].flatten(-2)
		x = self.input(x).relu()
		x = self.hidden(x)
		x = self.output(x)
		return x