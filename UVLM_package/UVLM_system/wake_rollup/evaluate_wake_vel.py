from os import name
from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size
from UVLM_package.UVLM_system.solve_circulations.biot_savart_comp_vc_temp import BiotSvart
from UVLM_package.UVLM_system.solve_circulations.induced_velocity_comp import InducedVelocity


class EvalWakeVel(Model):
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

        wake_vortex_pts_shapes = [
            tuple((nt, item[1], 3)) for item in surface_shapes
        ]  # nt,ny,3
        bdnwake_shapes = [
            (x[0] + y[0] - 1, x[1], 3)
            for x, y in zip(surface_shapes, wake_vortex_pts_shapes)
        ]
        wake_vel_shapes = [(x[0] * x[1], 3) for x in wake_vortex_pts_shapes]

        self.add(BiotSvart(
            eval_pt_names=wake_coords_names,
            vortex_coords_names=bdnwake_coords_names,
            eval_pt_shapes=wake_vortex_pts_shapes,
            vortex_coords_shapes=bdnwake_shapes,
            output_names=output_names,
            circulation_names=circulation_names,
            vc=True,
        ),
                 name='evaluate_wake_aics')

        # TODO: fixed this later after seperation of circulations
        aic_shapes = [(x[0] * x[1] * (y[0] - 1) * (y[1] - 1), 3)
                      for x, y in zip(wake_vortex_pts_shapes, bdnwake_shapes)]
        circulations_shapes = [
            ((x[0] - 1) * (x[1] - 1) + (y[0] - 1) * (y[1] - 1))
            for x, y in zip(surface_shapes, wake_vortex_pts_shapes)
        ]
        self.add(InducedVelocity(aic_names=output_names,
                                 circulation_names=circulation_names,
                                 aic_shapes=aic_shapes,
                                 circulations_shapes=circulations_shapes,
                                 v_induced_names=v_induced_wake_names),
                 name='evaluate_wake_ind_vel')

        kinematic_vel_names = [
            x + '_kinematic_vel' for x in self.parameters['surface_names']
        ]
        # TODO: check this part for the whole model
        model_wake_total_vel = Model()
        for i in range(len(v_induced_wake_names)):
            v_induced_wake_name = v_induced_wake_names[i]
            wake_vel_shape = wake_vel_shapes[i]
            wake_vortex_pts_shape = wake_vortex_pts_shapes[i]
            kinematic_vel_name = kinematic_vel_names[i]
            v_induced_wake = model_wake_total_vel.declare_variable(
                v_induced_wake_name, shape=wake_vel_shape)
            kinematic_vel = model_wake_total_vel.declare_variable(
                kinematic_vel_name, shape=wake_vel_shape)
            v_total_wake = csdl.reshape((v_induced_wake + kinematic_vel),
                                        new_shape=(wake_vortex_pts_shape))
            model_wake_total_vel.register_output(v_total_wake_names[i],
                                                 v_total_wake)
        self.add(model_wake_total_vel, name='wake_total_vel')


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

    model_1.add(BdnWakeCombine(
        surface_names=surface_names,
        surface_shapes=surface_shapes,
        nt=nt,
    ),
                name='BdnWakeCombine')

    model_1.add(EvalWakeVel(
        surface_names=surface_names,
        surface_shapes=surface_shapes,
        nt=nt,
    ),
                name='EvalWakeVel')
    sim = Simulator(model_1)
    sim.run()
    sim.visualize_implementation()

    v_induced_names = [x + '_wake_induced_vel' for x in surface_names]
    # print('gamma_b', gamma_b.shape, gamma_b)
    for i in range(len(surface_shapes)):
        v_induced_name = v_induced_names[i]
        # surface_gamma_b_name = surface_names[i] + '_gamma_b'

        print(v_induced_name, sim[v_induced_name].shape, sim[v_induced_name])
