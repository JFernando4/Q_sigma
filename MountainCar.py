from EnvironmentBase import EnvironmentBase
from pylab import random, cos

'''
mountain car (not modified to have an extra penalty for non-zero action)
'''


class MountainCar(EnvironmentBase):
    def init_state(self):
        position = -0.6 + random() * 0.2
        return position, 0.0

    def do_action(self, S, A):
        position, velocity = S
        if not A in (0, 1, 2):
            print('Invalid action:', A)
            raise Exception
        R = -1  # if A==1 else -1.5
        A -= 1
        velocity += 0.001 * A - 0.0025 * cos(3 * position)
        if velocity < -0.07:
            velocity = -0.07
        elif velocity >= 0.07:
            velocity = 0.06999999
        position += velocity
        if position >= 0.5:
            return R, None
        if position < -1.2:
            # position = -1.2
            # velocity = 0.0
            position = -0.6 + random() * 0.2
            velocity = 0.0
            R = -100
        return R, (position, velocity)

    def get_num_actions(self):
        return 3
