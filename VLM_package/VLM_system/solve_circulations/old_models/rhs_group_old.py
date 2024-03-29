# from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size

from VLM_package.VLM_system.solve_circulations.kinematic_velocity_comp import KinematicVelocityComp
from VLM_package.VLM_system.solve_circulations.assemble_aic import AssembleAic
from VLM_package.VLM_system.solve_circulations.compute_normal_comp import ComputeNormal
from VLM_package.VLM_system.solve_circulations.projection_comp import Projection


class RHS(Model):
    """
    Compute various geometric properties for VLM analysis.
    These are used primarily to help compute postprocessing quantities,
    such as wave CD, viscous CD, etc.
    Some of the quantities, like `normals`, are used to compute the RHS
    of the AIC linear system.
    A \gamma_b + b + M \gamma_w = 0
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
        self.parameters.declare('n_wake_pts_chord', default=5)
        self.parameters.declare('method',
                                values=['fw_euler', 'bk_euler'],
                                default='bk_euler')
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('bd_vortex_shapes', types=list)
        self.parameters.declare('delta_t')

    def define(self):
        n_wake_pts_chord = self.parameters['n_wake_pts_chord']
        delta_t = self.parameters['delta_t']
        bd_vortex_shapes = self.parameters['bd_vortex_shapes']
        surface_names = self.parameters['surface_names']

        bd_vtx_coords_names = [x + '_bd_vtx_coords' for x in surface_names]
        bd_vtx_normal_names = [x + '_bd_vtx_normals' for x in surface_names]
        coll_pts_coords_names = [x + '_coll_pts_coords' for x in surface_names]
        wake_coords_names = [x + '_wake_coords' for x in surface_names]
        bd_coll_pts_shapes = [
            tuple(map(lambda i, j: i - j, item, (1, 1, 0)))
            for item in bd_vortex_shapes
        ]
        bd_normal_shape = bd_coll_pts_shapes
        wake_vortex_pts_shapes = [
            tuple((n_wake_pts_chord, item[1], item[2]))
            for item in bd_vortex_shapes
        ]

        for i in range(len(bd_vortex_shapes)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
        method = self.parameters['method']
        '''1. project the kinematic velocity on to the bd_vertices'''
        frame_vel = self.declare_variable('frame_vel')
        # bd_vortex_coords = self.declare_variable('bd_vortex_coords',
        #                                          shape=(nx, ny, 3))
        # coll_coords = self.declare_variable('coll_coords',
        #                                     shape=((nx - 1), (ny - 1), 3))

        m = KinematicVelocityComp(
            surface_names=surface_names,
            surface_shapes=bd_vortex_shapes,  # (2*3,3)
        )
        self.add(m, name='KinematicVelocityComp')

        m = ComputeNormal(
            vortex_coords_names=bd_vtx_coords_names,
            normals_names=bd_vtx_normal_names,
            vortex_coords_shapes=bd_vortex_shapes,
        )
        self.add(m, name='ComputeNormal')  # shape=(2,3,3)

        kinematic_vel_names = [
            x + '_kinematic_vel' for x in self.parameters['surface_names']
        ]
        kinematic_vel_shapes = [
            tuple((item[0] * item[1], item[2])) for item in bd_coll_pts_shapes
        ]

        m = Projection(
            input_vel_names=kinematic_vel_names,
            normal_names=bd_vtx_normal_names,
            output_vel_names='b',  # this is b
            input_vel_shapes=kinematic_vel_shapes,  #rotatonal_vel_shapes
            normal_shapes=bd_coll_pts_shapes,
        )
        self.add(m, name='Projection_k_vel')
        '''2. compute M (bk_euler) or M\gamma_w (fw_euler)'''

        wake_coords_reshaped_names = [
            x + '_wake_coords_reshaped' for x in surface_names
        ]
        for i in range(len(surface_names)):
            wake_coords_reshaped_name = wake_coords_reshaped_names[i]
            ny = bd_vortex_shapes[i][1]
            wake_coords = self.declare_variable(wake_coords_names[i],
                                                shape=(1, n_wake_pts_chord, ny,
                                                       3))
            wake_coords_reshaped = csdl.reshape(wake_coords,
                                                (n_wake_pts_chord, ny, 3))
            self.register_output(wake_coords_reshaped_name,
                                 wake_coords_reshaped)

        m = AssembleAic(
            bd_coll_pts_names=coll_pts_coords_names,
            wake_vortex_pts_names=wake_coords_reshaped_names,
            bd_coll_pts_shapes=bd_coll_pts_shapes,
            wake_vortex_pts_shapes=wake_vortex_pts_shapes,
            full_aic_name='aic_M',
            delta_t=delta_t,  # one line of wake vortex for fix wake
        )
        self.add(m, name='AssembleAic')
        '''3. compute the size of the full AIC (coll_pts_coords_names, wake_coords_names) matrix'''

        aic_shape_row = aic_shape_col = 0

        for i in range(len(bd_coll_pts_shapes)):
            aic_shape_row += (bd_coll_pts_shapes[i][0] *
                              bd_coll_pts_shapes[i][1])
            aic_shape_col += ((wake_vortex_pts_shapes[i][0] - 1) *
                              (wake_vortex_pts_shapes[i][1] - 1))

        # print('aic_M-----------', (aic_shape_row, aic_shape_col, 3))
        '''3. project the aic on to the bd_vertices'''
        m = Projection(
            input_vel_names=['aic_M'],
            normal_names=bd_vtx_normal_names,
            output_vel_names='M_mat',  # this is b
            input_vel_shapes=[(aic_shape_row, aic_shape_col, 3)
                              ],  #rotatonal_vel_shapes
            normal_shapes=bd_coll_pts_shapes)  # NOTE: need to fix this later
        self.add(m, name='Projection_aic')


if __name__ == "__main__":

    def generate_simple_mesh(nx, ny, n_wake_pts_chord=None):
        if n_wake_pts_chord == None:
            mesh = np.zeros((nx, ny, 3))
            mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))
            mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
            mesh[:, :, 2] = 0.
        else:
            mesh = np.zeros((n_wake_pts_chord, nx, ny, 3))
            for i in range(n_wake_pts_chord):
                mesh[i, :, :, 0] = np.outer(np.arange(nx), np.ones(ny))
                mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
                mesh[i, :, :, 2] = 0.
        return mesh

    model_1 = Model()

    frame_vel_val = np.array([1, 0, 1])
    bd_vortex_coords_val = generate_simple_mesh(3, 4)
    wake_coords_val = np.array([
        [2., 0., 0.],
        [2., 1., 0.],
        [2., 2., 0.],
        [2., 3., 0.],
        [42., 0., 0.],
        [42., 1., 0.],
        [42., 2., 0.],
        [42., 3., 0.],
    ]).reshape(2, 4, 3)
    # coll_val = np.random.random((4, 5, 3))

    frame_vel = model_1.create_input('frame_vel', val=frame_vel_val)
    bd_vortex_coords = model_1.create_input('bd_vortex_coords',
                                            val=bd_vortex_coords_val)
    wake_coords = model_1.create_input('wake_coords', val=wake_coords_val)
    nx = 3
    ny = 4
    coll_pts = 0.25 * (bd_vortex_coords[0:nx-1, 0:ny-1, :] +\
                                               bd_vortex_coords[0:nx-1, 1:ny, :] +\
                                               bd_vortex_coords[1:, 0:ny-1, :]+\
                                               bd_vortex_coords[1:, 1:, :])
    model_1.register_output('coll_coords', coll_pts)
    model_1.add(RHS())

    sim = Simulator(model_1)
    sim.run()
    sim.visualize_implementation()
    # print('aic is', sim['aic'])
    # print('v_ind is', sim['v_ind'].shape, sim['v_ind'])
