import gym
from gym import spaces
from gym.envs.classic_control import rendering
import numpy as np
from coord import Coord
from colors import *
from maps import *

class GridWorld(gym.Env):
	metadata = {'render.modes': ['human', 'rgb_array']}
	
	def __init__(self, map, n_actions, random):
		assert n_actions in [9, 18], "The magnitude of the action space must be either 9 or 18"

		self.n_actions = n_actions
		self.random = random
		
		self.blockers = map.blockers
		self.size = map.size
		self.objects = map.objects
		self.max_steps = map.max_steps
		
		self.rows, self.cols = self.size
		self.p = self.rows * self.cols
		self.m = len(self.objects)
		self.N = max([obj.i for obj in self.objects]) + 1
		
		self.n_states = self.p * 2 ** self.m
		
		self.observation_space = spaces.Discrete(self.n_states)
		self.action_space = spaces.Discrete(self.n_actions)
		
		self.blocker_squares = [self.tuple2int(*t) for t in np.vstack(np.where(self.blockers)).T]
		self.square2blocker = {square:int(square in self.blocker_squares) for square in range(self.p)}
		
		self.viewer = None
		self.screen_width = 500
		self.screen_height = int(self.rows / self.cols * self.screen_width)
		self.tile_separation = 0.05
		self.tile_width = self.screen_width / ((1 + self.tile_separation) * self.cols + 2 + self.tile_separation)
		self.tile_height = self.screen_height / ((1 + self.tile_separation) * self.rows + 2 + self.tile_separation)
		
	def coord2int(self, coord: Coord):
		return coord.y * self.cols + coord.x
		
	def tuple2int(self, x, y):
		return y * self.cols + x
		
	def int2coord(self, i):
		return Coord(*self.int2tuple(i))
		
	def int2tuple(self, i):
		return (i % self.cols, i // self.cols)
		
	def get_unblocked_squares(self, n=1):
		blocked = np.array(list(self.square2blocker.values()))
		unblocked = np.nonzero(1 - blocked)[0]
		return np.random.choice(unblocked, replace = False, size = n)
		
	def get_empty_squares(self, n=1):
		blocked = np.array(list(self.square2blocker.values()))
		rewarded = np.array(list(self.square2reward.values()))
		empty = np.where(blocked + rewarded == 0)[0]
		return np.random.choice(empty, replace = False, size = n)
		
	def is_coord_bounded(self, coord: Coord):
		return coord.x >= 0 and coord.x < self.cols and coord.y >= 0 and coord.y < self.rows
	
	def is_tuple_bounded(self, x, y):
		return x >= 0 and x < self.cols and y >= 0 and y < self.rows
		
	def collect_beneath(self, coord: Coord):
		position = self.coord2int(coord)
		reward, terminal = 0, False
		
		for i, square in enumerate(self.reward_squares):
			if square == position:
				reward, terminal = self.objects[i].collect()
				
				if self.random and reward:
					if not self.viewer is None:
						x, y = self.int2tuple(self.reward_squares[i])
						
						l = (1 + self.tile_separation) * self.tile_width * (x + 1)
						r = (1 + self.tile_separation) * self.tile_width * (x + 2) - self.tile_separation * self.tile_width
						b = (1 + self.tile_separation) * self.tile_height * (y + 1)
						t = (1 + self.tile_separation) * self.tile_height * (y + 2) - self.tile_separation * self.tile_height
					
						empty_tile = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
						empty_tile.set_color(*white)
						self.viewer.add_geom(empty_tile)
						
					self.square2reward[self.reward_squares[i]] = 0
					self.reward_squares[i] = self.get_empty_squares()
					self.square2reward[self.reward_squares[i]] = 1
					
				break
				
		return reward, terminal
		
	def is_blocked(self, coord):
		if isinstance(coord, Coord):
			return self.square2blocker[self.coord2int(coord)]
		elif isinstance(coord, tuple):
			return self.square2blocker[self.tuple2int(*coord)]
		else:
			raise Exception("is_blocked only accepts a Coord or a 2D tuple as input")
		
	
	def get_state(self):
		if self.random:
			state_tensor = np.zeros((self.N, self.rows, self.cols))
			
			for y in range(self.rows):
				for x in range(self.cols):
					position = self.tuple2int(x, y)
					
					if self.square2reward[position]:
						idx = self.reward_squares.tolist().index(position)
						state_tensor[:, y, x] = np.eye(self.N)[self.objects[idx].i]
					else:
						state_tensor[:, y, x] = np.zeros(self.N)
						
			return state_tensor
		else:
			return sum([int(obj.hidden) * 2 ** (i + 1) for i, obj in enumerate(self.objects)]) // 2 + self.coord2int(self.state)
		
	def reset(self):
		squares = self.get_unblocked_squares(n = self.m + 1)
		
		self.reward_squares = squares[:self.m]
		self.state = self.int2coord(squares[-1])
		
		self.square2reward = {square:0 for square in range(self.p)}
		for square in self.reward_squares:
			self.square2reward[square] = 1
			
		self.timestep = 0
		self.episode_reward = 0
		self.action_history = []
		self.reward_history = []
		self.terminal = False
		
		return self.get_state()
		
	def step(self, action):
		self.timestep += 1
		self.action_history.append(action)
		
		for obj in self.objects:
			obj.update()
			
		movement_action = action % 9
		
		if movement_action == 0:
			movement_state = self.state + Coord(-1, -1)
		elif movement_action == 1:
			movement_state = self.state + Coord(0, -1)
		elif movement_action == 2:
			movement_state = self.state + Coord(0, 1)
		elif movement_action == 3:
			movement_state = self.state + Coord(-1, 0)
		elif movement_action == 4:
			movement_state = self.state
		elif movement_action == 5:
			movement_state = self.state + Coord(1, 0)
		elif movement_action == 6:
			movement_state = self.state + Coord(-1, 1)
		elif movement_action == 7:
			movement_state = self.state + Coord(0, 1)
		elif movement_action == 8:
			movement_state = self.state + Coord(1, 1)
			
		reward = 0
		
		if self.is_coord_bounded(movement_state) and not self.is_blocked(movement_state):
			if self.n_actions == 9:
				self.state = movement_state
				reward, terminal = self.collect_beneath(self.state)
			else:
				if action < 9:
					self.state = movement_state
				else:
					reward, terminal = self.collect_beneath(movement_state)
					
		self.episode_reward += reward
		self.reward_history.append(reward)
		self.terminal |= self.timestep >= self.max_steps
		
		return self.get_state(), reward, self.terminal, {}
		
	def render(self, mode='human'):		
		if self.viewer is None:
			self.viewer = rendering.Viewer(self.screen_width, self.screen_height)
			
			background = rendering.FilledPolygon([(0, 0), (0, self.screen_height), (self.screen_width, self.screen_height), (self.screen_width, 0)])
			background.set_color(*maroon)
			self.viewer.add_geom(background)
			
			for y in range(self.rows):
				for x in range(self.cols):
					l = (1 + self.tile_separation) * self.tile_width * (x + 1)
					r = (1 + self.tile_separation) * self.tile_width * (x + 2) - self.tile_separation * self.tile_width
					b = (1 + self.tile_separation) * self.tile_height * (y + 1)
					t = (1 + self.tile_separation) * self.tile_height * (y + 2) - self.tile_separation * self.tile_height
					
					tile = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
					
					if self.square2blocker[self.tuple2int(x, y)]:
						tile.set_color(*maroon)
					else:
						tile.set_color(*white)
						
					self.viewer.add_geom(tile)
				
			agent_tile = rendering.FilledPolygon([(0, 0), (0, self.tile_height), (self.tile_width, self.tile_height), (self.tile_width, 0)])
			agent_tile.set_color(*black)
			self.agent_translation = rendering.Transform()
			agent_tile.add_attr(self.agent_translation)
			self.viewer.add_geom(agent_tile)
		
		for pos, obj in zip(self.reward_squares, self.objects):
			x, y = self.int2tuple(pos)
				
			l = (1 + self.tile_separation) * self.tile_width * (x + 1)
			r = (1 + self.tile_separation) * self.tile_width * (x + 2) - self.tile_separation * self.tile_width
			b = (1 + self.tile_separation) * self.tile_height * (y + 1)
			t = (1 + self.tile_separation) * self.tile_height * (y + 2) - self.tile_separation * self.tile_height
				
			reward_tile = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
			
			if obj.hidden:
				reward_tile.set_color(*white)
			else:
				reward_tile.set_color(*obj.color)
				
			self.viewer.add_geom(reward_tile)
			
		agent_x = (1 + self.tile_separation) * self.tile_width * (self.state.x + 1)
		agent_y = (1 + self.tile_separation) * self.tile_height * (self.state.y + 1)
		self.agent_translation.set_translation(agent_x, agent_y)
				
		return self.viewer.render(return_rgb_array = mode == 'rgb_array')
		
	def close(self):
		if not self.viewer is None:
			self.viewer.close()
			self.viewer = None
			
from tqdm import tqdm
import cv2

map = Small()
env = GridWorld(map, 9, True)

episodes = 10000
for episode in range(episodes):
	state = env.reset()
	terminal = False
	while not terminal:
		action = env.action_space.sample()
		state, reward, terminal, info = env.step(action)
		
		for i, n in enumerate(np.uint8(state)):
			cv2.imshow(str(i), cv2.resize(n * 255, tuple(v * 35 for v in env.size[::-1]), interpolation = cv2.INTER_AREA))
			cv2.waitKey(1)
			
		env.render()
	
env.close()
