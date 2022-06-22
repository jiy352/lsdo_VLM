from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size


def generate_simple_mesh(nx, ny, n_wake_pts_chord=None, offset=0):
    if n_wake_pts_chord == None:
        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))
        mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T + offset
        mesh[:, :, 2] = 0.
    else:
        mesh = np.zeros((n_wake_pts_chord, nx, ny, 3))
        for i in range(n_wake_pts_chord):
            mesh[i, :, :, 0] = np.outer(np.arange(nx), np.ones(ny))
            mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T + offset
            mesh[i, :, :, 2] = 0.
    return mesh


if __name__ == "__main__":
    pass