import gym
import numpy as np

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
import tensorflow as tf

id = "Taxi-v3"
env = gym.make(id)

print ("""
------------------------------
 {id}
------------------------------
 Observation | {observation}
 Action      | {action}
------------------------------
""".format(id = id,
           observation = env.observation_space,
           action = env.action_space))

class Agent(Model):
    def __init__(self, num_states, num_actions, m=30):
        """
        Args:
            num_states   the magnitude of the discrete state space
            num_actions  the magnitude of the discrete action space
            m            the dimensionality of the categorical prediction vector, y
        """
        super(Agent, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.m = m

        self.project_upwards = Dense(64, activation = 'relu')
        self.hidden1 = Dense(64, activation = 'relu')
        self.hidden2 = Dense(64, activation = 'relu')
        self.linear1 = Dense(32, activation = 'relu')
        self.linear2 = Dense(32, activation = 'relu')
        self.pi_linear = Dense(num_actions)
        self.y_linear = Dense(m)

    def call(self, state):
        """
        Args:
            state  integer observation
        Returns:
            pi     q values for each valid action
            y      a categorical prediction vector of dimensionality m
        """
        x = self.project_upwards(state)
        x = self.hidden1(x)
        x = self.hidden2(x)
        a = self.linear1(x)
        b = self.linear2(x)
        pi = self.pi_linear(a)
        y = self.y_linear(b)
        return pi, y

class LPG(Model):
    def __init__(self, lstm_units=256, m=30):
        """
        Args:
            lstm_units  the output dimensionality of the LSTM layer
            m           the dimensionality of the categorical prediction vector, y
        """
        super(LPG, self).__init__()
        self.lstm_units = lstm_units
        self.m = m

        self.lstm = LSTM(lstm_units, go_backwards = True)
        self.hidden = Dense(128, activation = 'relu')
        self.linear1 = Dense(64, activation = 'relu')
        self.linear2 = Dense(64, activation = 'relu')
        self.pi_hat_linear = Dense(1)
        self.y_hat_linear = Dense(m)

    def call(self, x):
        """
        Inputs a list of transitions represented by vectors (r, d, γ, π(a|s), y0, y1) for each
        timestep t, where r is a reward, d is a binary value indicating episode-termination,
        gamma is a discount factor, pi is the probability of the chosen action, and y0 and y1 are
        the categorical prediction vectors of the current state and next state, respectively.

        The LPG architecture first passes the two subsequent categorical prediction vectors through a
        shared embedding network, which is simply comprised of a Dense layer with 16 rectified linear units,
        followed by a Dense layer with one rectified linear unit. This mapping converts the categorical
        prediction vectors into scalar values so that they conform to the LSTM input shape of [N, 6], where
        N is the length of the episode in timesteps.

        Given this input of N transition vectors, the LSTM layer outputs a vector of length 256 by default.
        The output of the LSTM is split down two alternate sequences of densely-connected layers. The final outputs
        of the LPG network are pi hat and y hat, where pi hat specified how the action-probability should be adjusted,
        and y hat specifies a target categorical distribution that the agent should predict for a given state. y hat does
        not have a relevant impact upon the policy until the LPG discovers useful semantics (e.g., value functions) of it
        and uses y to indirectly change the policy by bootstrapping.

        The framework, however, is not restricted to this particular form of agent update and
        architecture (e.g., categorical prediction with KL-divergence). This particular agent update rule is derived from
        the update rule in the REINFORCE algorithm. Other modifications are feasible, but the authors explore this specific form
        partly inspired by the success of Distributional RL (refer to sources below).

        [1] M. G. Bellemare, W. Dabney, and R. Munos. A distributional perspective on reinforcement
            learning. In Proceedings of the 34th International Conference on Machine Learning-Volume 70,
            pages 449–458. JMLR. org, 2017.
        [6] W. Dabney, M. Rowland, M. G. Bellemare, and R. Munos. Distributional reinforcement learning
            with quantile regression. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.

        Args:
            x  a list of transitions
        """
        o = self.lstm(x)
        o = self.hidden(o)
        a = self.linear1(o)
        b = self.linear2(o)
        pi_hat = self.pi_hat_linear(a)
        y_hat = self.y_hat_linear(b)
        return pi_hat, y_hat

class HyperparameterGroup:
    """
    A pair of agent hyperparameters, denoted by alpha.

    Consists of alpha_lr and alpha_y, where alpha_lr denotes a learning rate
    and alpha_y denotes a coefficient for the prediction update (see Eq. (2))
    """
    options = {'lr': [0.0005, 0.001, 0.002, 0.005],
               'y': [0.1, 0.5, 1]}

    def __init__(self):
        self.lr = np.random.choice(self.options['lr'])
        self.y = np.random.choice(self.options['y'])

class Experience:
    def __init__(self, env: gym.Env, agent: Agent, hyperparams: HyperparameterGroup):
        """
        Args:
            env          an instance of an environment (denoted Ɛ), sampled from the environment distribution
            agent        an agent (denoted θ), whose parameters are sampled from an initial agent parameter distribution
            hyperparams  a pair of agent hyperparameters (denoted α), sampled from a hyperparameter sampling distribution
        """
        self.env = env
        self.agent = agent
        self.hyperparams = hyperparams
