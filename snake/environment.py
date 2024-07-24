from collections import deque
from math import floor, ceil
from random import randint
import torch

class SnakeEnv:
	C_EMPTY, C_WALL, C_FOOD, C_HEAD, C_BODY_UP, C_BODY_RIGHT, C_BODY_DOWN, C_BODY_LEFT = 0, 1, 2, 3, 4, 5, 6, 7
	D_UP, D_RIGHT, D_DOWN, D_LEFT = 0, 1, 2, 3
	R_LOSE, R_FOOD, R_WIN, R_TRUNCATED, R_OTHER = -1, 1, 1, 0, 0

	def __init__(self, board_size, empty_size, max_turns, device=None):
		self.board_size = board_size
		self.empty_size = empty_size
		self.max_turns = max_turns
		self.device = device

	def reset(self):
		self.remaining_turns = self.max_turns
		self.board = torch.zeros((self.board_size, self.board_size), dtype=torch.int, device=self.device)
		side_wall_width = (self.board_size - self.empty_size) / 2
		if side_wall_width:
			self.board[:floor(side_wall_width)] = self.board[-ceil(side_wall_width):] = self.board[:, :floor(side_wall_width)] = self.board[:, -ceil(side_wall_width):] = SnakeEnv.C_WALL
		self.body = deque([(
			randint(floor(side_wall_width), self.board_size - ceil(side_wall_width) - 1),
			randint(floor(side_wall_width), self.board_size - ceil(side_wall_width) - 1)
		)])
		self.board[*self.body[0]] = SnakeEnv.C_HEAD
		self._place_food()
		return self.board

	def step(self, action):
		self.remaining_turns -= 1
		if self.remaining_turns == 0:
			return self.board, SnakeEnv.R_TRUNCATED, True
		if action == SnakeEnv.D_UP: new_head = (self.body[0][0] - 1, self.body[0][1])
		elif action == SnakeEnv.D_RIGHT: new_head = (self.body[0][0], self.body[0][1] + 1)
		elif action == SnakeEnv.D_DOWN: new_head = (self.body[0][0] + 1, self.body[0][1])
		elif action == SnakeEnv.D_LEFT: new_head = (self.body[0][0], self.body[0][1] - 1)
		if not (0 <= new_head[0] < self.board_size and 0 <= new_head[1] < self.board_size) or (self.board[*new_head] > SnakeEnv.C_HEAD or self.board[*new_head] == SnakeEnv.C_WALL) and new_head != self.body[-1]:
			reward, done = SnakeEnv.R_LOSE, True
		else:
			self.body.appendleft(new_head)
			if new_head == self.food:
				if len(self.body) == self.board_size * self.board_size:
					reward, done = SnakeEnv.R_WIN, True
				else:
					reward, done = SnakeEnv.R_FOOD, False
					self._place_food()
			else:
				self.board[*self.body[-1]] = SnakeEnv.C_EMPTY
				self.body.pop()
				reward, done = SnakeEnv.R_OTHER, False
			self.board[*new_head] = SnakeEnv.C_HEAD
			if len(self.body) > 1:
				self.board[*self.body[1]] = [SnakeEnv.C_BODY_UP, SnakeEnv.C_BODY_RIGHT, SnakeEnv.C_BODY_DOWN, SnakeEnv.C_BODY_LEFT][action]
		return self.board, reward, done

	def _place_food(self):
		empties = torch.nonzero(self.board == SnakeEnv.C_EMPTY)
		self.food = tuple(empties[randint(0, len(empties) - 1)])
		self.board[*self.food] = SnakeEnv.C_FOOD