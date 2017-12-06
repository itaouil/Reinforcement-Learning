#!/usr/bin/env/python
# -*- coding: utf-8 -*-

"""
    Reinforcement Learning.
"""

# Import modules
import math
import numpy as np

# Import config file
import config as cf

# Global variables
QVAL    = {}
STATES  = []

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

def create_domain():
    """
        The routine creates a 4 x 6 numpy
        matrix and intialises the various
        obstacles as well as the start and
        goal positions.

        Returns:
            numpy array: states
    """
    global STATES

    # Create initial states
    STATES = np.zeros((cf.data['X'], cf.data['Y']))

    # Add start, goal and obstacles
    STATES[0, 4] = 100
    STATES[1, 2] = -10
    STATES[1, 5] = -100
    STATES[2, 0] = -100
    STATES[3, 4] = -10

def create_qvalues():
    """
        The routine creates the qvalues numpy
        array which gets updated in the SARSA
        and QLEARNING algorithms.

        Returns:
            numpy array: qvalues
    """
    global QVAL

    # Create a dictionary contaning
    # our values for the SARSA and
    # QLEARNING algorithms
    for row in range(0, cf.data['X']):
        for col in range(0, cf.data['Y']):
            QVAL[str(row) + str(col)] = random(100, 200)

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
            param3: domain (game states)

        Returns:
            int: reward for the given state-action pair
    """
    # New state
    reward_state = env_move_det(s, a)
    return STATES[reward_state[0]][reward_state[1]]

def agt_choose(s, epsilon):
    """
        Performs an ε policy where
        with probability 1 - ε the
        optimal policy is chosen and
        where with ε a random action
        is chosen otherwise.

        Arguments:
            param1: currrent state
            param2: epsilon

        Returns:
            str: action to take
    """
    # Evaluates the greedy policy
    if (random(1, 11) / 10.0) < 1 - epsilon:

        # Policies array
        policies = []

        # Compute all possible policies
        # given the current state
        for a in cf.data['actions']:
            policies.append(env_reward(s, a))

        # Return best action
        return cf.data['actions'][policies.index(max(policies))]
    else:
        return cf.data['actions'][random(0, 4)]

def agt_learn_sarsa(alpha, s, a, r, next_s, next_a):
    """
        Sarsa algorithm implementation.

        Arguments:
            param1: alpha
            param2: current state
            param3: reward
            param4: next state
            param5: next action
    """
    # Convert current state to key (for dictionary check)
    key_curr = ''.join(s)
    key_next = ''.join(next_s)

    # Action index
    a_index = cf.data['actions'].index(a)

    # Update sarsa, baby !
    QVAL[key_curr][a_index] = (1 - alpha) * QVAL[key_curr][a_index] + alpha * (r + cf.data['gamma'] * QVAL[key_next][next_a])

def main():
    create_domain()
    create_qvalues()
    print("States and QValues created...")
    print("Action: Up", actions('up'))
    print("Deterministic state: ", env_move_det((0, 5), 'up'))
    print("Stochastic state: ", env_move_sto((0, 4), 'left'))
    print("Agent choose: ", agt_choose((1, 3), 0.3))

if __name__ == '__main__':
    main()
