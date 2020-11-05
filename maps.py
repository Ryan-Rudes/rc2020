from colors import *
from objects import *
colors = [blue, red, yellow]

class Map(object):
    def __init__(self, width, height, object_types, max_steps_per_episode):
        self.width = width
        self.height = height
        self.size = (width, height)
        self.p = width * height
        self.m = sum(list(object_types.keys()))
        self.n_states = self.p * 2 ** self.m
        self.max_steps_per_episode = max_steps_per_episode
        self.info = {'Number of actions': '9 or 18',
                     'Size': '{width} x {height}'.format(width = width, height = height),
                     'Objects': ', '.join(['[{}, {}, {}]'.format(*params) if N == 1 else '{} x [{}, {}, {}]'.format(N, *params) for N, params in object_types.items()]),
                     'Maximum steps per episode': str(max_steps_per_episode)}

class TabularMap(Map):
    def __init__(self, width, height, object_types, max_steps_per_episode):
        super(TabularMap, self).__init__(width, height, object_types, max_steps_per_episode)

        self.objects = [TabularObject(*params, color) for color, (N, params) in zip(colors, object_types.items()) for i in range(N)]
        self.info['Observaion'] = 'State index (integer)'

class RandomMap(Map):
    def __init__(self, width, height, object_types, max_steps_per_episode):
        super(RandomMap, self).__init__(width, height, object_types, max_steps_per_episode)

        self.objects = [RandomObject(*params, color) for color, (N, params) in zip(colors, object_types.items()) for i in range(N)]
        self.info['Observaion'] = '{0, 1}^(N x H x W)'

class TabularDense(TabularMap):
    def __init__(self):
        super(TabularDense, self).__init__(11, 11, {
            2: [1, 0, 0.05],
            1: [-1, 0.5, 0.1],
            1: [-1, 0, 0.5]
        }, 500)

class TabularSparse(TabularMap):
    def __init__(self):
        super(TabularSparse, self).__init__(13, 13, {
            1: [1, 1, 0],
            1: [-1, 1, 0]
        }, 50)

class TabularLongHorizon(TabularMap):
    def __init__(self):
        super(TabularLongHorizon, self).__init__(11, 11, {
            2: [1, 0, 0.01],
            2: [-1, 0.5, 1]
        }, 1000)

class TabularLongerHorizon(TabularMap):
    def __init__(self):
        super(TabularLongerHorizon, self).__init__(7, 9, {
            2: [1, 0.1, 0.01],
            2: [-1, 0.8, 1]
        }, 2000)

class TabularLongDense(TabularMap):
    def __init__(self):
        super(TabularLongDense, self).__init__(11, 11, {
            4: [1, 0, 0.005]
        }, 2000)

class RandomDense(RandomMap):
    def __init__(self):
        super(RandomDense, self).__init__(11, 11, {
            2: [1, 0, 0.05],
            1: [-1, 0.5, 0.1],
            1: [-1, 0, 0.5]
        }, 500)

class RandomLongHorizon(RandomMap):
    def __init__(self):
        super(RandomLongHorizon, self).__init__(11, 11, {
            2: [1, 0, 0.01],
            2: [-1, 0.5, 1]
        }, 1000)

class RandomSmall(RandomMap):
    def __init__(self):
        super(RandomSmall, self).__init__(5, 7, {
            2: [1, 0, 0.05],
            2: [-1, 0.5, 0.1]
        }, 500)

class RandomSmallSparse(RandomMap):
    def __init__(self):
        super(RandomSmallSparse, self).__init__(5, 7, {
            1: [1, 1, 1],
            2: [-1, 1, 1]
        }, 50)

class RandomVeryDense(RandomMap):
    def __init__(self):
        super(RandomVeryDense, self).__init__(11, 11, {
            1: [1, 0, 1]
        }, 2000)
