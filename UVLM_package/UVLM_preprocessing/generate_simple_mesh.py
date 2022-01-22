from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size


def generate_simple_mesh(nx, ny, nt=None, offset=0):
    if nt == None:
        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))
        mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T + offset
        mesh[:, :, 2] = 0.
    else:
        mesh = np.zeros((nt, nx, ny, 3))
        for i in range(nt):
            mesh[i, :, :, 0] = np.outer(np.arange(nx), np.ones(ny))
            mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T + offset
            mesh[i, :, :, 2] = 0.
    return mesh


def compute_wake_coords(nx, ny, nt, h_stepsize, frame_vel_val, offset=0):
    delta_t = h_stepsize

    wake_coords_val_x = np.einsum(
        'i,j->ij',
        (nx - 1 + 0.25 + delta_t * np.arange(nt) * (-frame_vel_val[0])),
        np.ones(ny),
    ).flatten()

    wake_coords_val_z = np.einsum(
        'i,j->ij',
        (0 + delta_t * np.arange(nt) * (-frame_vel_val[2])),
        np.ones(ny),
    ).flatten()

    wake_coords_val_y = (np.einsum(
        'i,j->ji',
        generate_simple_mesh(nx, ny)[-1, :, 1],
        np.ones(nt),
    ) + (delta_t * np.arange(nt) *
         (-frame_vel_val[1])).reshape(-1, 1)).flatten()
    mesh = generate_simple_mesh(nx, ny)
    wake_coords_val = np.zeros((nt, ny, 3))
    wake_coords_val[:, :, 0] = wake_coords_val_x.reshape(nt, ny)
    wake_coords_val[:, :, 1] = wake_coords_val_y.reshape(nt, ny) + offset
    wake_coords_val[:, :, 2] = wake_coords_val_z.reshape(nt, ny)
    return wake_coords_val


if __name__ == "__main__":
    nx = 2
    ny = 3
    nt = 4
    h_stepsize = 1
    frame_vel_val = np.array([-1, 0, -1])
    wake_coords_val = compute_wake_coords(nx, ny, nt, h_stepsize,
                                          frame_vel_val)
