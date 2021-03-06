#!/usr/bin/env/python
# -*- coding: utf-8 -*-

"""
    Reinforcement Learning.
"""

# Import modules
import math
import time
import numpy as np
import matplotlib.pyplot as plt

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
    STATES[0, 1] = 5
    STATES[1, 2] = -20
    STATES[1, 3] = 5
    STATES[1, 4] = 10
    STATES[1, 5] = -100
    STATES[2, 0] = -100
    STATES[2, 2] = -10
    STATES[2, 5] = -10
    STATES[3, 4] = -20

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
    for row in range(cf.data['X']):
        for col in range(cf.data['Y']):
            QVAL[str(row) + str(col)] = [150.0, 150.0, 150.0, 150.0]

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

def val_to_action(val):
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
        0 : 'up',
        1 : 'down',
        2 : 'right',
        3 : 'left'
    }[val]

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
    # New state
    if random(1, 11) / 10.0 <= 0.8:
        new_state = env_move_det(s, a)
    else:
        # Random ortogonal move
        new_action = get_ortogonal_move(a)[random(0, 2)]
        new_state = env_move_det(s, new_action)

    return new_state

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

    if s == reward_state:
        return 0
    else:
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
    key = str(s[0]) + str(s[1])

    # Evaluates the greedy policy
    if random(1, 11) / 10.0 <= 1 - epsilon:
        # Compute all possible policies
        # given the current state
        # Return best action
        return val_to_action(QVAL[key].index(max(QVAL[key])))
    else:
        return val_to_action(QVAL[key].index(QVAL[key][random(0, 4)]))

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
    key_curr = str(s[0]) + str(s[1])
    key_next = str(next_s[0]) + str(next_s[1])

    # Action index
    curr_a_index = cf.data['actions'].index(a)
    next_a_index = cf.data['actions'].index(next_a)

    # Update sarsa's values, baby !
    QVAL[key_curr][curr_a_index] = (1 - alpha) * QVAL[key_curr][curr_a_index] + alpha * (r + cf.data['gamma'] * QVAL[key_next][next_a_index])

def agt_learn_q(alpha, s, a, r, next_s):
    """
        QLEARNING algorithm implementation.

        Arguments:
            param1: alpha
            param2: current state
            param3: action
            param4: reward
            param5: next state
    """
    # Convert current state to key (for dictionary check)
    key_curr = str(s[0]) + str(s[1])
    key_next = str(next_s[0]) + str(next_s[1])

    # Action index
    a_index = cf.data['actions'].index(a)

    # Update qlearning's values, baby !
    QVAL[key_curr][a_index] = (1 - alpha) * QVAL[key_curr][a_index] + alpha * (r + cf.data['gamma'] * max(QVAL[key_next]))

def agt_learn_final(alpha, s, a, r):
    """
        Absorbing state evaluation

        Arguments:
            param1: alpha
            param2: current state
            param3: action
            param4: reward
    """
    # Convert current state to key (for dictionary check)
    key = str(s[0]) + str(s[1])

    # Action index
    a_index = cf.data['actions'].index(a)

    # Update
    QVAL[key][a_index] = (1 - alpha) * QVAL[key][a_index] + alpha * r

def agt_reset_value():
    """
        Resets the action-value function
        to random values.
    """
    create_qvalues()

def main():
    # Intialise domain
    # and state-action
    # value pairs
    create_domain()
    create_qvalues()

    # Clear rewards
    rewards = [0 for x in range(cf.data['episodes'])]

    for epoch in range(cf.data['epochs']):

        # Clean QLVAL dictionaries
        agt_reset_value()

        for episode in range(cf.data['episodes']):

            # Defining the type of learning
            learning = episode < cf.data['episodes'] - 50
            eps = cf.data['epsilon'] if learning else 0
            cumulative_gamma = 1

            # Get initial state and action
            s = (3, 0)
            a = agt_choose(s, eps)

            for timestep in range(cf.data['T']):
                # Get next state
                next_s = env_move_det(s, a)
                # next_s = env_move_sto(s, a)

                # Compute reward and add it
                r = env_reward(s, a)
                rewards[episode] += (cumulative_gamma * r) / cf.data['epochs']
                cumulative_gamma *= cf.data['gamma']
                next_a = agt_choose(next_s, eps)

                if learning:
                    if STATES[next_s[0]][next_s[1]] in [100, -100] or timestep == cf.data['T'] - 1:
                        # Run final and move to next episode
                        agt_learn_final(cf.data['alpha'], s, a, r)
                        break
                    else:
                        # agt_learn_q(cf.data['alpha'], s, a, r, next_s)
                        agt_learn_sarsa(cf.data['alpha'], s, a, r, next_s, next_a)

                a = next_a
                s = next_s

if __name__ == '__main__':
    main()
