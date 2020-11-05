from utils import pos_to_coords
import numpy as np

class Object(object):
    """An object (distributed across the map)"""

    def __init__(self, r, e_term, e_respawn, color):
        """
        Object constructor

        Args:
            r          reward received upon collection
            e_term     probability of episode termination upon collection
            e_respawn  probability of reappearance each timestep after collection
            color      rgb color tuoke of the object for rendering
        """

        # Object variables
        self.r = r
        self.e_term = e_term
        self.e_respawn = e_respawn

        # Rendering variables
        self.color = color

        # Whether or not the object is hidden
        self.hidden = False

    def _move(self, position, game):
        """
        Sets the location of the object

        Args:
            position  integer representation of a board position
        """

        # Ensure that the specified position is valid
        assert position < game.p

        self.position = position
        self.x, self.y = pos_to_coords(position, game.width)

    def _collect(self):
        """
        Collects the object, ensuring that it was visible beforehand

        An object may only be collected if it was already visible, otherwise,
        an error is thrown. The object disappears afterwards, and the episode
        terminates with probability e_term.

        Returns:
            a boolean indicating whether or not the episode has been terminated
            the reward of the collected object
        """

        # Ensure that the object being collected is currently visible
        assert not self.hidden, "Collected an object that was already hidden"

        # Hide the object after collection
        self.hidden = True

        return np.random.random() < self.e_term, self.r

    def update(self):
        """
        Performs one timestep update. If the object is hidden,
        it reappears with probability e_respawn.
        """

        if self.hidden and np.random.random() < self.e_respawn:
            self.hidden = False

class TabularObject(Object):
    def __init__(self, r, e_term, e_respawn, color):
        super(TabularObject, self).__init__(r, e_term, e_respawn, color)

    def collect(self):
        terminal, reward = self._collect()
        return terminal, reward

class RandomObject(Object):
    def __init__(self, r, e_term, e_respawn, color):
        super(RandomObject, self).__init__(r, e_term, e_respawn, color)

    def collect(self, game):
        terminal, reward = self._collect()
        self._move(game._random_empty_positions(), game)
        return terminal, reward
