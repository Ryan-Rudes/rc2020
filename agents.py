import numpy as np
from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def initialize(self, loc, scale):
        pass

    def random_action(self, s, ε):
        """
        Given parameter ε, samples a random action with probability ε.
        Otherwise, returns the best action according to the current policy
        and the specified state.
        """
        if np.random.random() < ε:
            return np.random.randint(0, self.A)
        else:
            return self.best_action(s)

    @abstractmethod
    def __call__(self, s):
        pass

class TabularAgent(Agent):
    """
    An agent class for Tabular Grid World, which does not
    require function approximation.
    """

    def __init__(self, S, A):
        """
        In Tabular Grid World environments, an agent is represented by a table with
        distinct π(a|s) and y(s) values for each state with no function approximation.
        Args:
            S: The number of states (p * 2^m in Tabular Grid World)
            A: The number of actions (either 9 or 18 in Grid World environments)
        """
        self.S = S
        self.A = A

    def initialize(self, loc=0, scale=1):
        """
        Initializes the agent with random state-action and y values, sampled
        from a normal distribution.

        Args:
            loc: Mean of the distribution
            scale: Standard deviation of the distribution
        """
        self.π = np.random.normal(loc=loc, scale=scale, size = (self.S, self.A))
        self.y = np.random.normal(loc=loc, scale=scale, size = (self.S,))

    def best_action(self, s):
        """
        Returns the optimal action according to the current policy
        and the specified state.
        """
        return np.argmax(self.π[s])

    def __call__(self, s):
        """
        Call method for the Agent class

        Returns:
            The state-action values for all actions from state s
            The y value of state s
        """
        return self.π[s], self.y[s]

class FunctionalAgent(Agent):
    """
    An agent class for Random Grid World, which implements
    linear function approximation.
    """

    def __init__(self, A, d):
        """
        In Random Grid World environments, the locations of objects are randomized within
        a lifetime, resulting in an exponentially larger state space. Function approximation
        is required to represent an agent for these environments. The observation consists of
        a binary tensor of size N×H×W, where N is the number of object types, and H, W are
        the height and width of the grid, respectively.

        Args:
            A: The number of actions (either 9 or 18 in Grid World environments)
            d: The dimensionality of the observation (N×H×W)
        """
        self.A = A
        self.d = d # N×H×W

    def initialize(self, loc=0, scale=1):
        """
        Initializes the agent's linear approximation function with random parameters, sampled
        from a normal distribution.

        Args:
            loc: Mean of the distribution
            scale: Standard deviation of the distribution
        """
        self.θ = np.random.normal(loc=loc, scale=scale, size = self.d)
        self.θ_zero = np.random.normal(loc=loc, scale=scale)

    def __call__(self, s):
        """
        Approximates the value function for the given state, using the parameters of the approximation
        function as coefficients in a weighted sum of these parameters and the observation features.
        """
        assert s.shape == (self.d,), "The observation provided is an incorrect shape"
        return self.θ_zero + sum([parameter * feature for parameter, feature in zip(self.θ, s)])

class BinaryAgent(TabularAgent):
    """
    A tabular agent class for Delayed Chain MDP
    """

    def __init__(self, S):
        """
        In Standard Delayed Chain MDP environments, the agent has a binary choice of actions for each timestep,
        The first action determines the reward at the end of the episode (1 or -1), and the rest of the actions
        are irrelevent. The goal is to determine whether the agent is capable of ignoring the excessive noise
        from the second action onwards and focus upon the initial action which indicates the terminal reward.
        In some Delayed Chain MDP environments, the agent also receives noisy rewards {1, -1} throughout the
        episode that are also entirely independent of the actions taken.

        Args:
            S: The maximum number of states for an MDP sampled in this environment (a.k.a. the maximum chain length)
        """
        super(BinaryAgent, self).__init__(S, 2)

    def initialize(self, loc=0, scale=1):
        self.π = np.random.normal(loc=loc, scale=scale, size = (self.S, self.A))

    def __call__(self, s):
        """
        Call method for the BinaryAgent class

        Returns:
            The state-action values for all actions from state s
        """
        return self.π[s]

# Demo for TabularAgent
agent = TabularAgent(100, 9)
agent.initialize()
print (agent.best_action(2))
print (agent(2))

# Demo for FunctionalAgent
agent = FunctionalAgent(9, 4, 10, 10)
agent.initialize()
print (agent(np.random.randint(0, 2, size = (400,))))

# Demo for BinaryAgent
agent = BinaryAgent(30)
agent.initialize()
print (agent.best_action(2))
print (agent(2))
