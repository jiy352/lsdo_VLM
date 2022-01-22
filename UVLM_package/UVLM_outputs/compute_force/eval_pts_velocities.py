from csdl_om import Simulator
from csdl import Model
import csdl
from matplotlib.pyplot import clabel
import numpy as np
from numpy.core.fromnumeric import size

from scipy.sparse import csc_matrix
from UVLM_package.UVLM_system.wake_rollup.combine_bd_wake_comp import BdnWakeCombine
from UVLM_package.UVLM_system.solve_circulations.biot_savart_comp_vc_temp import BiotSvart
from UVLM_package.UVLM_system.solve_circulations.induced_velocity_comp import InducedVelocity


class EvalPtsVel(Model):
    """
    Compute various geometric properties for VLM analysis.
    These are used primarily to help compute postprocessing quantities,
    such as wave CD, viscous CD, etc.
    Some of the quantities, like `normals`, are used to compute the RHS
    of the AIC linear system.
    A \gamma_b = b - M \gamma_w
    parameters
    ----------

    collocation_pts[num_vortex_panel_x*num_vortex_panel_y] : csdl array
        all the bd vertices collocation_pts     
    wake_pts[num_vortex_panel_x*num_vortex_panel_y] : csdl array
        all the wake panel collcation pts 
    wake_circulations[num_wake_panel] : csdl array
        a concatenate vector of the wake circulation strength
    Returns
    -------
    vel_col_w[num_evel_pts_x*num_vortex_panel_x* num_evel_pts_y*num_vortex_panel_y,3]
    csdl array
        the velocities computed using the aic_col_w from biot svart's law
        on bound vertices collcation pts induces by the wakes
    """
    def initialize(self):
        self.parameters.declare('eval_pts_names', types=list)
        self.parameters.declare('eval_pts_shapes', types=list)
        self.parameters.declare('eval_pts_location', default=0.25)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        # stands for quarter-chord
        self.parameters.declare('nt')
        self.parameters.declare('delta_t')

    def define(self):
        eval_pts_names = self.parameters['eval_pts_names']
        eval_pts_shapes = self.parameters['eval_pts_shapes']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        eval_pts_location = self.parameters['eval_pts_location']

        nt = self.parameters['nt']
        delta_t = self.parameters['delta_t']

        bdnwake_coords_names = [x + '_bdnwake_coords' for x in surface_names]

        wake_coords_reshaped_names = [
            x + '_wake_coords_reshaped' for x in surface_names
        ]

        wake_vortex_pts_shapes = [
            tuple((nt, item[1], 3)) for item in surface_shapes
        ]

        bdnwake_shapes = [
            (x[0] + y[0] - 1, x[1], 3)
            for x, y in zip(surface_shapes, wake_vortex_pts_shapes)
        ]
        output_names = [x + '_aic_force' for x in surface_names]
        circulation_names = [x + '_bdnwake_gamma' for x in surface_names]

        aic_shapes = [(x[0] * x[1] * (y[0] - 1) * (y[1] - 1), 3)
                      for x, y in zip(eval_pts_shapes, bdnwake_shapes)]

        circulations_shapes = [
            ((x[0] - 1) * (x[1] - 1) + (y[0] - 1) * (y[1] - 1))
            for x, y in zip(surface_shapes, wake_vortex_pts_shapes)
        ]
        v_induced_wake_names = [
            x + '_eval_pts_induced_vel' for x in surface_names
        ]
        v_total_eval_names = [x + '_eval_total_vel' for x in surface_names]
        # eval_pts_coords_shapes = [(x[0] - 1, x[1] - 1, 3)
        #                           for x in eval_pts_shapes]

        eval_vel_shapes = [(x[0] * x[1], 3) for x in eval_pts_shapes]
        # TODO: might change here for higher order numerical method
        n = 1
        ode_surface_shapes = [(n, ) + item for item in surface_shapes]

        #!TODO!: rewrite this comp for mls
        for i in range(len(eval_pts_names)):
            mesh = self.declare_variable(surface_names[i],
                                         shape=ode_surface_shapes[i])
            bdnwake = self.declare_variable(bdnwake_coords_names[i],
                                            surface_shapes[i])
            nx = surface_shapes[i][0]
            ny = surface_shapes[i][1]
            # print(mesh.shape)
            eval_pts_coords = (
                (1 - eval_pts_location) * 0.5 * mesh[:, 0:-1, 0:-1, :] +
                (1 - eval_pts_location) * 0.5 * mesh[:, 0:-1, 1:, :] +
                eval_pts_location * 0.5 * mesh[:, 1:, 0:-1, :] +
                eval_pts_location * 0.5 * mesh[:, 1:, 1:, :])
            self.register_output(
                'eval_pts_coords',
                csdl.reshape(eval_pts_coords, (nx - 1, ny - 1, 3)))

        self.add(BdnWakeCombine(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            nt=nt,
        ),
                 name='BdnWakeCombine')

        #!TODO:fix this for mls
        for i in range(len(surface_shapes)):
            nx = surface_shapes[i][0]
            ny = surface_shapes[i][1]
            bdnwake_coords = self.declare_variable(bdnwake_coords_names[i],
                                                   shape=(nt + nx - 1, ny, 3))

        self.add(BiotSvart(
            eval_pt_names=['eval_pts_coords'],
            vortex_coords_names=bdnwake_coords_names,
            eval_pt_shapes=eval_pts_shapes,
            vortex_coords_shapes=bdnwake_shapes,
            output_names=output_names,
            circulation_names=circulation_names,
            delta_t=delta_t,
            vc=True,
        ),
                 name='eval_pts_aics')
        # print('bsafter')
        # print('eval_pt_names', ['eval_pts_coords'])
        # print('vortex_coords_names', bdnwake_coords_names)
        # print('eval_pt_shapes', eval_pts_shapes)
        # print('vortex_coords_shapes', bdnwake_shapes)
        # print('circulation_names', circulation_names)
        # print('delta_t', delta_t)

        self.add(InducedVelocity(aic_names=output_names,
                                 circulation_names=circulation_names,
                                 aic_shapes=aic_shapes,
                                 circulations_shapes=circulations_shapes,
                                 v_induced_names=v_induced_wake_names),
                 name='eval_pts_ind_vel')

        # kinematic_vel_names = [
        #     x + '_kinematic_vel' for x in self.parameters['surface_names']
        # ]
        # TODO: check this part for the whole model
        model_wake_total_vel = Model()
        for i in range(len(v_induced_wake_names)):
            v_induced_wake_name = v_induced_wake_names[i]
            eval_vel_shape = eval_vel_shapes[i]

            wake_vortex_pts_shape = wake_vortex_pts_shapes[i]
            # kinematic_vel_name = kinematic_vel_names[i]

            v_induced_wake = model_wake_total_vel.declare_variable(
                v_induced_wake_name, shape=eval_vel_shape)
            # print('v_induced_wake shape=======================',
            #       v_induced_wake.shape)
            # !!TODO!! this needs to be fixed for more general cases to compute the undisturbed vel

            # kinematic_vel = model_wake_total_vel.declare_variable(
            #     kinematic_vel_name, shape=wake_vel_shape)
            frame_vel = model_wake_total_vel.declare_variable('frame_vel',
                                                              shape=(3, ))
            frame_vel_expand = csdl.expand(frame_vel,
                                           eval_vel_shape,
                                           indices='i->ji')

            v_total_wake = csdl.reshape((v_induced_wake + frame_vel_expand),
                                        new_shape=eval_vel_shape)

            model_wake_total_vel.register_output(v_total_eval_names[i],
                                                 v_total_wake)
        # print('name_in_eval_vel', v_total_eval_names[i])

        self.add(model_wake_total_vel, name='eval_pts_total_vel')


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

    surface_names = ['wing']
    surface_shapes = [(3, 4, 3)]
    eval_pts_names = ['force_pts']
    eval_pts_shapes = [(2, 3, 3)]
    delta_t = 1

    nt = 4
    model_1 = Model()
    frame_vel_val = np.random.random((3, ))
    force_pts = model_1.create_input('wing', val=generate_simple_mesh(3, 4))
    # force_pts = model_1.create_input('force_pts',
    #                                  val=np.random.random(eval_pts_shapes[0]))

    model_1.add(EvalPtsVel(
        eval_pts_names=eval_pts_names,
        eval_pts_shapes=eval_pts_shapes,
        surface_names=surface_names,
        surface_shapes=surface_shapes,
        nt=nt,
        delta_t=delta_t,
    ),
                name='EvalWakeVel')
    print('bs3')

    sim = Simulator(model_1)
    sim.run()
    print('bs4')
    sim.visualize_implementation()

    # v_induced_names = [x + '_wake_induced_vel' for x in surface_names]
    # # print('gamma_b', gamma_b.shape, gamma_b)
    # for i in range(len(surface_shapes)):
    #     v_induced_name = v_induced_names[i]
    #     # surface_gamma_b_name = surface_names[i] + '_gamma_b'

    #     print(v_induced_name, sim[v_induced_name].shape, sim[v_induced_name])