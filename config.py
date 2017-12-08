#!/usr/bin/env/python
# -*- coding: utf-8 -*-

"""
    The following script stores
    the configuration parameters.
"""

# Import packages
import math
import numpy as np

data = {

    # Grid boundaries
    'X'         : 4,
    'Y'         : 6,

    # Set of actions
    'actions'   : ['up', 'down', 'right', 'left'],

    # Gamma value
    'gamma'     : 0.99,

    # Epochs
    'epochs'     : 20,

    # Episodes
    'episodes'  : 500,

    'epsilon'   : 0.1,

    'alpha'     : 0.1,

    'T'         : 48

}
