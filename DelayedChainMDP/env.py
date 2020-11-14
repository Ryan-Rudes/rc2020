import gym
from gym import spaces
from gym.envs.classic_control import rendering
import numpy as np

class Map(object):
    metadata = {'observation.types': ['integer', 'bit_array']}

    def __init__(self, chain_length_range, noisy_rewards, observation_type='integer'):
        assert observation_type in self.metadata['observation.types']
        self.min_len, self.max_len = chain_length_range
        self.noisy_rewards = noisy_rewards
        self.observation_type = observation_type

class DelayedChainMDP(gym.Env):
    """Delayed Chain MDP environment"""
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, map):
        super(DelayedChainMDP, self).__init__()

        self.map = map
        self.chain_length = np.random.randint(self.map.min_len, self.map.max_len + 1)
        self.observation_space = spaces.Discrete(self.chain_length)
        self.action_space = spaces.Discrete(2)
        self.info = {'Observation': 'State index (integer)',
                     'Number of actions': 2,
                     'Chain length': self.chain_length,
                     'Noisy rewards': self.map.noisy_rewards}

        self.viewer = None

    def _observe(self):
        if self.map.observation_type == 'integer':
            return self.timestep
        else:
            if self.timestep == 0:
                return None
            else:
                self.bits = np.hstack((np.random.randint(0, 2, size = 20), np.array([self.rewarded_action, self.action == self.rewarded_action])))
                np.random.shuffle(self.bits)
                return self.bits

    def reset(self):
        self.rewarded_action = env.action_space.sample()
        self.timestep = 0
        self.close()
        return self._observe()

    def step(self, action):
        if self.timestep == 0:
            self.terminal_reward = (1 if action == 0 else -1)

        self.action = action
        self.timestep += 1
        self.terminal = self.timestep == self.chain_length
        self.reward = (self.terminal_reward if terminal else np.random.choice([-1, 1]) if self.map.noisy_rewards else 0)
        return self._observe(), self.reward, self.terminal, self.info

    def render(self, mode='human'):
        radius = 20
        diameter = radius * 2

        separation_factor = 1.2
        separation = int(diameter * separation_factor)

        left_padding = radius + separation
        screen_width = left_padding + (self.chain_length + 2) * separation + radius
        screen_height = 500

        resolution = 300

        y_center = screen_height // 2
        y_top = y_center + radius * separation_factor
        y_bottom = y_center - radius * separation_factor

        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            initial_circle = rendering.make_circle(radius = radius, res = resolution)
            translation = rendering.Transform(translation = (left_padding, y_center))
            initial_circle.add_attr(translation)
            self.viewer.add_geom(initial_circle)

            self.viewer.add_geom(rendering.Line((left_padding + radius, y_center), (left_padding + separation // 2, y_center)))
            self.viewer.add_geom(rendering.Line((left_padding + separation // 2, y_bottom), (left_padding + separation // 2, y_top)))
            self.viewer.add_geom(rendering.Line((left_padding + separation // 2, y_top), (left_padding + separation - radius, y_top)))
            self.viewer.add_geom(rendering.Line((left_padding + separation // 2, y_bottom), (left_padding + separation - radius, y_bottom)))

            self.circles = []
            self.translations = []
            x_offset = left_padding + separation

            for t in range(self.chain_length - 1):
                top_circle = rendering.make_circle(radius = radius, res = resolution, filled = False)
                bottom_circle = rendering.make_circle(radius = radius, res = resolution, filled = False)

                top_translation = rendering.Transform(translation = (x_offset, y_top))
                bottom_translation = rendering.Transform(translation = (x_offset, y_bottom))

                top_circle.add_attr(top_translation)
                bottom_circle.add_attr(bottom_translation)

                self.viewer.add_geom(top_circle)
                self.viewer.add_geom(bottom_circle)

                top_line = rendering.Line((x_offset + radius, y_top), (x_offset + separation - radius, y_top))
                bottom_line = rendering.Line((x_offset + radius, y_bottom), (x_offset + separation - radius, y_bottom))

                self.viewer.add_geom(top_line)
                self.viewer.add_geom(bottom_line)

                if self.rewarded_action == 1:
                    self.circles.append(top_circle)
                    self.translations.append(top_translation)
                else:
                    self.circles.append(bottom_circle)
                    self.translations.append(bottom_translation)

                x_offset += separation

            terminal_top_circle = rendering.make_circle(radius = radius, res = resolution)
            terminal_bottom_circle = rendering.make_circle(radius = radius, res = resolution)

            terminal_top_translation = rendering.Transform(translation = (x_offset, y_top))
            terminal_bottom_translation = rendering.Transform(translation = (x_offset, y_bottom))

            terminal_top_circle.add_attr(terminal_top_translation)
            terminal_bottom_circle.add_attr(terminal_bottom_translation)

            terminal_top_circle.set_color(52/255, 235/255, 95/255) # green
            terminal_bottom_circle.set_color(235/255, 64/255, 52/255) # red

            self.viewer.add_geom(terminal_top_circle)
            self.viewer.add_geom(terminal_bottom_circle)

        if self.timestep < self.chain_length:
            filled_circle = self.viewer.draw_circle(radius = radius, res = resolution)
            stroke_circle = self.viewer.draw_circle(radius = radius, res = resolution, filled = False)
            if self.reward == 1:
                filled_circle.set_color(52/255, 235/255, 95/255) # green
            elif self.reward == -1:
                filled_circle.set_color(235/255, 64/255, 52/255) # red
            filled_circle.add_attr(self.translations[self.timestep - 1])
            stroke_circle.add_attr(self.translations[self.timestep - 1])
            self.viewer.add_geom(filled_circle)
            self.viewer.add_geom(stroke_circle)

        return self.viewer.render(return_rgb_array = mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

Short = Map([5, 30], False)
ShortAndNoisy = Map([5, 30], True)
Long = Map([5, 50], False)
LongAndNoisy = Map([5, 50], True)
StateDistraction = Map([5, 30], False, observation_type = 'bit_array')

from tqdm import tqdm

env = DelayedChainMDP(ShortAndNoisy)
episodes = 10

for episode in tqdm(range(episodes), 'Episode'):
    terminal = False
    state = env.reset()
    while not terminal:
        action = env.action_space.sample()
        state, reward, terminal, info = env.step(action)
        env.render()

env.close()
