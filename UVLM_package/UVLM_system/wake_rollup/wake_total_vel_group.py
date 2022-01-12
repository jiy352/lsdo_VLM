from os import name
from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size

from UVLM_package.UVLM_system.wake_rollup.combine_bd_wake_comp import BdnWakeCombine
from UVLM_package.UVLM_system.wake_rollup.evaluate_wake_vel import EvalWakeVel
from UVLM_package.UVLM_system.wake_rollup.combine_bd_wake_comp import BdnWakeCombine


class WakeTotalVel(Model):
    """
    compute induced velocity of the wake points
    wake_vel = BS(wake_coords, bsnwake) @ gamma_w
    parameters
    ----------
    
    wake_coords
    bdnwake_coords
    gamma's - gamma_b and gamma_w (seperated) 

    Returns
    -------
    wake_induced_vel[nt, num_evel_pts_x, num_evel_pts_y, 3] : csdl array
    """
    def initialize(self):

        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('nt')

    def define(self):
        # add_input name and shapes
        nt = self.parameters['nt']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        wake_coords_names = [x + '_wake_coords' for x in surface_names]
        bdnwake_coords_names = [x + '_bdnwake_coords' for x in surface_names]
        output_names = [x + '_aic_wake' for x in surface_names]
        circulation_names = [x + '_bdnwake_gamma' for x in surface_names]
        v_induced_wake_names = [x + '_wake_induced_vel' for x in surface_names]
        v_total_wake_names = [x + '_wake_total_vel' for x in surface_names]

        self.add(BdnWakeCombine(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            nt=nt,
        ),
                 name='BdnWakeCombine')

        self.add(EvalWakeVel(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            nt=nt,
        ),
                 name='EvalWakeVel')


if __name__ == "__main__":
    from UVLM_package.UVLM_system.wake_rollup.combine_bd_wake_comp import BdnWakeCombine

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

    model_1.add(WakeTotalVel(
        surface_names=surface_names,
        surface_shapes=surface_shapes,
        nt=nt,
    ),
                name='WakeTotalVels')

    sim = Simulator(model_1)
    sim.run()
    sim.visualize_implementation()

    v_induced_names = [x + '_wake_induced_vel' for x in surface_names]
    # print('gamma_b', gamma_b.shape, gamma_b)
    for i in range(len(surface_shapes)):
        v_induced_name = v_induced_names[i]
        # surface_gamma_b_name = surface_names[i] + '_gamma_b'

        print(v_induced_name, sim[v_induced_name].shape, sim[v_induced_name])
