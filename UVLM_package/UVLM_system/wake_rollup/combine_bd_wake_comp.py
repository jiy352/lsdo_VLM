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
        nt = self.parameters['nt']

        for i in range(len(surface_shapes)):
            bd_vxt_coords_name = surface_names[i] + '_bd_vtx_coords'
            wake_coords_name = surface_names[i] + '_wake_coords'
            bd_n_wake_coords_name = surface_names[i] + '_bdnwake_coords'

            surface_gamma_b_name = surface_names[i] + '_gamma_b'
            surface_gamma_w_name = surface_names[i] + '_gamma_w'
            bd_n_wake_circulation_name = surface_names[i] + '_bdnwake_gamma'

            surface_shape = surface_shapes[i]

            nx = surface_shapes[i][0]
            ny = surface_shapes[i][1]

            wake_shape = (nt, ) + (ny, 3)
            surface_gamma_b_shape = ((nx - 1) * (ny - 1))
            surface_gamma_w_shape = ((nt - 1) * (ny - 1))

            # add_input name and shapes
            bd_vxt_coords = self.declare_variable(bd_vxt_coords_name,
                                                  shape=surface_shape)
            wake_coords = self.declare_variable(wake_coords_name,
                                                shape=wake_shape)
            # add_input name and shapes
            surface_gamma_b = self.declare_variable(
                surface_gamma_b_name, shape=surface_gamma_b_shape)
            surface_gamma_w = self.declare_variable(
                surface_gamma_w_name, shape=surface_gamma_w_shape)
            # compute output bd_n_wake_coords
            bd_n_wake_coords = self.create_output(bd_n_wake_coords_name,
                                                  shape=(nx + nt - 1, ny, 3))
            bd_n_wake_coords[:nx, :, :] = bd_vxt_coords
            bd_n_wake_coords[nx:, :, :] = wake_coords[1:, :, :]
            # compute output bd_n_wake_gamma
            bd_n_wake_gamma = self.create_output(bd_n_wake_circulation_name,
                                                 shape=(surface_gamma_b_shape +
                                                        surface_gamma_w_shape))
            bd_n_wake_gamma[:surface_gamma_b_shape] = surface_gamma_b
            bd_n_wake_gamma[surface_gamma_b_shape:] = surface_gamma_w


if __name__ == "__main__":

    def generate_simple_mesh(nx, ny, nt=None, delta_y=0):
        if nt == None:
            mesh = np.zeros((nx, ny, 3))
            mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny)) + delta_y
            mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
            mesh[:, :, 2] = 0.
        else:
            mesh = np.zeros((nt, nx, ny, 3))
            for i in range(nt):
                mesh[i, :, :,
                     0] = np.outer(np.arange(nx), np.ones(ny)) + delta_y
                mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
                mesh[i, :, :, 2] = 0.
        return mesh

    surface_names = ['a', 'b']
    surface_shapes = [(3, 2, 3), (2, 4, 3)]
    nt = 2
    model_1 = Model()
    frame_vel_val = np.random.random((3, ))
    for i in range(2):
        bd_vxt_coords_name = surface_names[i] + '_bd_vtx_coords'
        wake_coords_name = surface_names[i] + '_wake_coords'
        surface_gamma_b_name = surface_names[i] + '_gamma_b'
        surface_gamma_w_name = surface_names[i] + '_gamma_w'

        nx = surface_shapes[i][0]
        ny = surface_shapes[i][1]

        surface_gamma_b_shape = ((nx - 1) * (ny - 1))
        surface_gamma_w_shape = ((nt - 1) * (ny - 1))

        bd_vxt_coords = model_1.create_input(bd_vxt_coords_name,
                                             val=generate_simple_mesh(
                                                 surface_shapes[i][0],
                                                 surface_shapes[i][1]))
        wake_coords = model_1.create_input(
            wake_coords_name,
            val=generate_simple_mesh(nt,
                                     surface_shapes[i][1],
                                     delta_y=surface_shapes[i][0] - 1))

        surface_gamma_b = model_1.create_input(
            surface_gamma_b_name, val=np.random.random(surface_gamma_b_shape))

        surface_gamma_w = model_1.create_input(
            surface_gamma_w_name, val=np.random.random(surface_gamma_w_shape))

    model_1.add(BdnWakeCombine(
        surface_names=surface_names,
        surface_shapes=surface_shapes,
        nt=nt,
    ),
                name='BdnWakeCombine')
    sim = Simulator(model_1)
    sim.run()

    for i in range(2):
        bd_vxt_coords_name = surface_names[i] + '_bd_vtx_coords'
        wake_coords_name = surface_names[i] + '_wake_coords'
        bd_n_wake_coords_name = surface_names[i] + '_bdnwake_coords'
        bd_n_wake_circulation_name = surface_names[i] + '_bdnwake_gamma'
        surface_gamma_b_name = surface_names[i] + '_gamma_b'
        surface_gamma_w_name = surface_names[i] + '_gamma_w'

        print(bd_vxt_coords_name, sim[bd_vxt_coords_name].shape,
              sim[bd_vxt_coords_name])
        print(wake_coords_name, sim[wake_coords_name].shape,
              sim[wake_coords_name])
        print(bd_n_wake_coords_name, sim[bd_n_wake_coords_name].shape,
              sim[bd_n_wake_coords_name])

        print(surface_gamma_b_name, sim[surface_gamma_b_name].shape,
              sim[surface_gamma_b_name])
        print(surface_gamma_w_name, sim[surface_gamma_w_name].shape,
              sim[surface_gamma_w_name])

        print(bd_n_wake_circulation_name,
              sim[bd_n_wake_circulation_name].shape,
              sim[bd_n_wake_circulation_name])
