"""
    Reinforcement Learning.
"""

# Import modules
import math
import numpy as np

# Import config file
import config as cf

""" Environment implementation """

def random(low, high):
    """
        Returns a random number
        between a low and hig.

        Arguments:
            param1: low boundary
            param2: high boundary

        Returns:
            int: random number between low inclusive and high exclusive
    """
    return np.random.randint(low, high)

def get_ortogonal_move(a):
    """
        It gives back the two
        ortogonal moves for the
        robot.

        Arguments:
            param1: robot's action

        Returns:
            list: possible robot's ortogonal moves
    """
    return {
        'up'    : ['left', 'right'],
        'down'  : ['left', 'right'],
        'right' : ['up', 'down'],
        'left'  : ['up', 'down']
    }[a]

def states():
    """
        The routine creates a 4 x 6 numpy
        matrix and intialises the various
        obstacles as well as the start and
        goal positions.

        Returns:
            numpy array: states
    """
    # Create initial states
    states = np.zeros((cf.data['X'], cf.data['Y']))

    # Add start, goal and obstacles
    states[0, 4] = 100
    states[1, 2] = -10
    states[1, 5] = -100
    states[2, 0] = -100
    states[3, 4] = -10

    return states

def actions(a):
    """
        The function acts as a
        lookup to decide how to
        move the robot in the
        grid depending on which
        move the robot makes.

        Arguments:
            param1: robot's action

        Returns:
            int: value of the movement in the array
    """
    return {
        'up'    : (-1, 0),
        'down'  : (1, 0),
        'right' : (0, 1),
        'left'  : (0, -1)
    }[a]

def env_move_det(s, a):
    """
        Deterministic movement where the
        robot moves with the given action
        if this doesn't go out of bound.

        Arguments:
            param1: current state (position of the robot in the map)
            param2: action of the robot

        Returns:
            tuple: robot's current x and y position in the grid
    """
    # Get state movement
    move = actions(a)

    # New state
    new_state = (s[0] + move[0], s[1] + move[1])

    # Check that move does not overflow
    # and return it, otherwise return the
    # current state
    if new_state[0] in range(cf.data['X']) and new_state[1] in range(cf.data['Y']):
        return new_state
    else:
        return s

def env_move_sto(s, a):
    """
        Stochastic movement where the
        robot moves with the given action
        with a 0.8 probability and with a
        0.1 probability with an ortogonal
        action.

        Arguments:
            param1: current state (position of the robot in the map)
            param2: action of the robot

        Returns:
            tuple: robot's current x and y position in the grid
    """
    # Get state movement
    move = actions(a)

    # Probabilities
    prob, ort_prob = random(1, 11) / 10.0, random(0, 1)

    # New state
    if prob <= 0.8:
        new_state = (s[0] + move[0], s[1] + move[1])
    else:
        # Random ortogonal move
        new_action = get_ortogonal_move(a)[random(0, 2)]
        new_move = actions(new_action)
        new_state = (s[0] + new_move[0], s[1] + new_move[1])

    # Check that move does not overflow
    # and return it, otherwise return the
    # current state
    if new_state[0] in range(cf.data['X']) and new_state[1] in range(cf.data['Y']):
        return new_state
    else:
        return s

def env_reward(s, a):
    """
        Calculates the reward for
        the current state and action.

        Arguments:
            param1: current state
            param2: action

        Returns:
            int: reward for the given state-action pair
    """
    # Move
    move = actions(a)

    # New state
    reward_state = (s[0] + move[0], s[1] + move[1])

    # Check if action in the state is valid
    if reward_state[0] in range(cf.data['X']) and reward_state[1] in range(cf.data['Y']):
        return states[reward_state[0], reward_state[1]]

def main():
    print("States:", states())
    print("Action: Up", actions('up'))
    print("Deterministic state: ", env_move_det((0, 5), 'up'))
    print("Stochastic state: ", env_move_det((0, 4), 'left'))

if __name__ == '__main__':
    main()