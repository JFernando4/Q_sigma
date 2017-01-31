from FunctionApproximatorBase import FunctionApproximatorBase
from pylab import random, asarray
from Tilecoder3 import IHT, tiles

"""
Tile Coder for Action Value Functions in the Mountain Car environment
"""


class TileCoderApproximatorAV(FunctionApproximatorBase):

    def __init__(self, numTilings=8, numActions=3):
        self.actions = numActions
        self.numTilings = numTilings
        self.alpha_factor = 1/self.numTilings
        self.numTiles = (self.numTilings**3)*4
        self.iht = IHT(self.numTiles)
        self.theta = 0.001 * random(self.numTiles * self.actions)
        super().__init__()

    def get_value(self, state, action):
        tile_indices = asarray(
            tiles(self.iht, self.numTilings, [8 * state[0] / (0.5 + 1.2), 8 * state[1] / (0.07 + 0.07)]),
            dtype=int) + (action * self.numTiles)
        return sum(self.theta[tile_indices])

    def update(self, state, action, value):
        tile_indices = asarray(
            tiles(self.iht, self.numTilings, [8*state[0]/(0.5+1.2), 8*state[1]/(0.07+0.07)]),
            dtype=int) + (action * self.numTiles)
        for index in tile_indices:
            self.theta[index] += value

    def get_theta(self):
        return self.theta

    def set_theta(self, new_theta):
        self.theta = new_theta

    def get_alpha_factor(self):
        return self.alpha_factor
