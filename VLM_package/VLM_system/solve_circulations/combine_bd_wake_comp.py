from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import shape, size


class BdnWakeCombine(Model):
    """
    combine bd and wake coords for the BS evaluation of the wake rollup
    wake_vel = BS(wake_coords, bsnwake) @ gamma_w

    parameters
    ----------
S

    angular_vel[1,] rad/sec
    bd_vtx_coords[num_evel_pts_x, num_evel_pts_y, 3] : csdl array 

    Returns
    -------
    kinematic_vel[nt, num_evel_pts_x, num_evel_pts_y, 3] : csdl array
        Induced velocities at found along the 3/4 chord.
    """
    def initialize(self):

        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('nt')

    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']

        num_nodes = surface_shapes[0][0]

        nt = self.parameters['nt']

        for i in range(len(surface_shapes)):
            bd_vxt_coords_name = surface_names[i] + '_bd_vtx_coords'
            wake_coords_name = surface_names[i] + '_wake_coords'
            bd_n_wake_coords_name = surface_names[i] + '_bdnwake_coords'

            surface_gamma_b_name = surface_names[i] + '_gamma_b'
            surface_gamma_w_name = surface_names[i] + '_gamma_w'
            bd_n_wake_circulation_name = surface_names[i] + '_bdnwake_gamma'

            surface_shape = surface_shapes[i]

            nx = surface_shapes[i][1]
            ny = surface_shapes[i][2]

            wake_shape = (num_nodes, nt, ny, 3)
            surface_gamma_b_shape = (num_nodes, (nx - 1) * (ny - 1))
            surface_gamma_w_shape = (num_nodes, (nt - 1), (ny - 1))

            # add_input name and shapes
            bd_vxt_coords = self.declare_variable(bd_vxt_coords_name,
                                                  shape=surface_shape)
            wake_coords = self.declare_variable(wake_coords_name,
                                                shape=wake_shape)
            # add_input name and shapes
            surface_gamma_b = self.declare_variable(
                surface_gamma_b_name, shape=surface_gamma_b_shape)
            surface_gamma_w = surface_gamma_b[:,
                                              (nx - 1) * (ny - 1) - (ny - 1):]
            # compute output bd_n_wake_coords
            bd_n_wake_coords = self.create_output(bd_n_wake_coords_name,
                                                  shape=(num_nodes,
                                                         nx + nt - 1, ny, 3))
            bd_n_wake_coords[:, :nx, :, :] = bd_vxt_coords
            bd_n_wake_coords[:, nx:, :, :] = wake_coords[:, 1:, :, :]

            # compute output bd_n_wake_gamma
            bd_n_wake_gamma = self.create_output(
                bd_n_wake_circulation_name,
                shape=(num_nodes,
                       surface_gamma_b_shape[1] + (nt - 1) * (ny - 1)))
            # print('combine bd wake surface_gamma_b shape',
            #       surface_gamma_b.shape)
            # print('combine bd wake surface_gamma_w shape',
            #       surface_gamma_w.shape)
            bd_n_wake_gamma[:, :surface_gamma_b_shape[1]] = surface_gamma_b
            bd_n_wake_gamma[:, surface_gamma_b_shape[1]:] = surface_gamma_w


if __name__ == "__main__":

    pass