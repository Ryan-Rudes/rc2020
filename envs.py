import gym
from gym import spaces
from gym.envs.classic_control import rendering

import numpy as np
import cv2

from objects import *
from maps import *
from colors import *
from utils import *

class GridWorld(object):
    """Grid World environment"""

    def __init__(self, n_actions):
        """
        GridWorld constructor

        Args:
            map        string identifier of a particular map
            n_actions  an integer, either 9 or 18
        """

        assert n_actions == 9 or n_actions == 18
        super(GridWorld, self).__init__()

        self.n_actions = n_actions
        self.action_space = spaces.Discrete(n_actions)
        self.map = map
        self.viewer = None

    def _get_image(self, width=256, height=256):
        tile_width = width // self.width
        tile_height = height // self.height
        image = np.full((tile_height * self.height, tile_width * self.width, 3), 255, dtype = np.uint8)

        image = cv2.rectangle(image, (self.x * tile_width, self.y * tile_height), ((self.x + 1) * tile_width, (self.y + 1) * tile_height), black, -1)
        for obj in self.objects:
            if not obj.hidden:
                image = cv2.rectangle(image, (obj.x * tile_width, obj.y * tile_height), ((obj.x + 1) * tile_width, (obj.y + 1) * tile_height), obj.color, -1)

        return image

    def _random_empty_positions(self, k=1):
        return np.random.choice(np.arange(self.p), size = k, replace = False)

    def _update(self):
        """
        Updates all objects in the world. Also, increments the timestep.

        Returns:
            boolean, indicating whether or not the episode is terminal due to maximum timesteps reached
        """

        for obj in self.objects:
            obj.update()

        self.timestep += 1
        return self.timestep >= self.max_steps_per_episode

    def _object_beneath(self, position):
        """Checks if an object is beneath a position. If so, returns the object."""

        for obj in self.objects:
            if obj.position == position:
                return obj

    def reset(self):
        """
        Resets the environment

        Returns:
            initial observation
        """

        # Select an empty position for each object
        object_positions = self._random_empty_positions(self.m)

        for position, obj in zip(object_positions, self.objects):
            obj._move(position, self)

        self.position = self._random_empty_positions()
        self.x, self.y = pos_to_coords(self.position, self.width)
        self.episode_reward = 0
        self.timestep = 0

        return self._get_state()

    def step(self, action):
        """
        Steps forwards an action in the environment

        Actions:
            0  UP_LEFT
            1  UP
            2  UP_RIGHT
            3  LEFT
            4  NONE
            5  RIGHT
            6  DOWN_LEFT
            7  DOWN
            8  DOWN_RIGHT

            9  COLLECT_UP_LEFT
            10 COLLECT_UP
            11 COLLECT_UP_RIGHT
            12 COLLECT_LEFT
            13 NONE
            14 COLLECT_RIGHT
            15 COLLECT_DOWN_LEFT
            16 COLLECT_DOWN
            17 COLLECT_DOWN_RIGHT
        """

        object_beneath = None

        if action == 0:
            if self.y > 0 and self.x > 0:
                self.x -= 1
                self.y -= 1
                self.position -= self.width + 1
        elif action == 1:
            if self.y > 0:
                self.y -= 1
                self.position -= self.width
        elif action == 2:
            if self.y > 0 and self.x < self.width - 1:
                self.x += 1
                self.y -= 1
                self.position -= self.width - 1
        elif action == 3:
            if self.x > 0:
                self.x -= 1
                self.position -= 1
        elif action == 4:
            pass
        elif action == 5:
            if self.x < self.width - 1:
                self.x += 1
                self.position += 1
        elif action == 6:
            if self.x > 0 and self.y < self.height - 1:
                self.x -= 1
                self.y += 1
                self.position += self.width - 1
        elif action == 7:
            if self.y < self.height - 1:
                self.y += 1
                self.position += self.width
        elif action == 8:
            if self.x < self.width - 1 and self.y < self.height - 1:
                self.x += 1
                self.y += 1
                self.position += self.width + 1

        if self.n_actions == 9:
            object_beneath = self._object_beneath(self.position)
        else:
            if action == 9:
                if self.y > 0 and self.x > 0:
                    object_beneath = self._object_beneath(self.position - (self.width + 1))
            elif action == 10:
                if self.y > 0:
                    object_beneath = self._object_beneath(self.position - self.width)
            elif action == 11:
                if self.y > 0 and self.x < self.width - 1:
                    object_beneath = self._object_beneath(self.position - (self.width - 1))
            elif action == 12:
                if self.x > 0:
                    object_beneath = self._object_beneath(self.position - 1)
            elif action == 13:
                object_beneath = self._object_beneath(self.position)
            elif action == 14:
                if self.x < self.width - 1:
                    object_beneath = self._object_beneath(self.position + 1)
            elif action == 15:
                if self.x > 0 and self.y < self.height - 1:
                    object_beneath = self._object_beneath(self.position + (self.width - 1))
            elif action == 16:
                if self.y < self.height - 1:
                    object_beneath = self._object_beneath(self.position + self.width)
            elif action == 17:
                if self.x < self.width - 1 and self.y < self.height - 1:
                    object_beneath = self._object_beneath(self.position + (self.width + 1))

        if not (object_beneath is None or object_beneath.hidden):
            terminal, reward = object_beneath.collect(self)
            self.episode_reward += reward
        else:
            terminal, reward = False, 0

        terminal |= self._update()

        return self._get_state(), reward, terminal, self.info

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()

        image = self._get_image()

        if mode == 'human':
            self.viewer.imshow(image)
        elif mode == 'rgb_array':
            return image
        else:
            raise NotImplementedError("Unknown render mode. Use either 'human' or 'rgb_array'.")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class TabularGridWorld(GridWorld):
    """Tabular Grid World environment"""

    metadata = {'render.modes': ['human', 'rgb_array'],
                'map.names': ['dense', 'sparse', 'long_horizon', 'longer_horizon', 'long_dense']}

    def __init__(self, n_actions):
        super(TabularGridWorld, self).__init__(n_actions)
        self.observation_space = spaces.Discrete(self.n_states)

    def _get_state(self):
        """Returns the current state in its integer representation"""
        return sum([int(obj.hidden) * 2 ** (i + 1) for i, obj in enumerate(self.objects)]) // 2 + self.position
class RandomGridWorld(GridWorld):
    """Random Grid World environment"""

    metadata = {'render.moves': ['human', 'rgb_array'],
                'map.names': ['dense', 'long_horizon', 'small', 'small_sparse', 'very_dense']}

    def __init__(self, n_actions):
        super(RandomGridWorld, self).__init__(n_actions)
        self.observation_space = spaces.Box(low = 0, high = 1, shape = (self.m, self.height, self.width), dtype = np.uint8)

    def _get_state(self):
        """Returns the current state in its integer representation"""
        color_set = list(set([tuple(object.color) for object in self.objects]))
        n_object_types = len(color_set)
        color2int = {color:i for i, color in enumerate(color_set)}
        state_tensor = np.zeros((n_object_types, self.height, self.width))

        for y in range(self.height):
            for x in range(self.width):
                position = y * self.width + x
                object_beneath = self._object_beneath(position)
                if object_beneath is None:
                    state_tensor[:, y, x] = np.zeros(n_object_types)
                else:
                    state_tensor[:, y, x] = np.eye(n_object_types)[color2int[tuple(object_beneath.color)]]

        return state_tensor

class TabularDenseGridWorld(TabularGridWorld, TabularDense):
    def __init__(self, n_actions):
        super().__init__(n_actions)
class TabularSparseGridWorld(TabularGridWorld, TabularSparse):
    def __init__(self, n_actions):
        super().__init__(n_actions)
class TabularLongHorizonGridWorld(TabularGridWorld, TabularLongHorizon):
    def __init__(self, n_actions):
        super().__init__(n_actions)
class TabularLongerHorizonGridWorld(TabularGridWorld, TabularLongerHorizon):
    def __init__(self, n_actions):
        super().__init__(n_actions)
class TabularLongDenseGridWorld(TabularGridWorld, TabularLongDense):
    def __init__(self, n_actions):
        super().__init__(n_actions)

class RandomDenseGridWorld(RandomGridWorld, RandomDense):
    def __init__(self, n_actions):
        super().__init__(n_actions)
class RandomLongHorizonGridWorld(RandomGridWorld, RandomLongHorizon):
    def __init__(self, n_actions):
        super().__init__(n_actions)
class RandomSmallGridWorld(RandomGridWorld, RandomSmall):
    def __init__(self, n_actions):
        super().__init__(n_actions)
class RandomSmallSparseGridWorld(RandomGridWorld, RandomSmallSparse):
    def __init__(self, n_actions):
        super().__init__(n_actions)
class RandomVeryDenseGridWorld(RandomGridWorld, RandomVeryDense):
    def __init__(self, n_actions):
        super().__init__(n_actions)

from tqdm import tqdm
env = RandomVeryDenseGridWorld(18)
episodes = 1000
for episode in tqdm(range(episodes), "Episode"):
    env.reset()
    terminal = False
    while not terminal:
        action = env.action_space.sample()
        obs, reward, terminal, info = env.step(action)
        env.render()
env.close()
