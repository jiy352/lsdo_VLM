from csdl import Model, ImplicitOperation, ScipyKrylov, NewtonSolver, NonlinearBlockGS
import numpy as np
from csdl_om import Simulator
import csdl

from VLM_package.VLM_system.solve_circulations.rhs_group_old import RHS
from VLM_package.VLM_system.solve_circulations.assemble_aic import AssembleAic
from VLM_package.VLM_system.solve_circulations.projection_comp import Projection
from VLM_package.VLM_system.solve_circulations.seperate_gamma_b import SeperateGammab


class SolveMatrix(Model):
    """
    Solve the AIC linear system to obtain the vortex ring circulations.
    A \gamma_b + b + M \gamma_w = 0

    A        size: (A_row, A_col)
        A_row = sum((nx[i] - 1) * (ny[i] - 1))
        A_col = sum((nx[i] - 1) * (ny[i] - 1))
    \gamma_b size: sum((nx[i] - 1) * (ny[i] - 1))
    b        size: sum((nx[i] - 1) * (ny[i] - 1))
    M        size: 
        M_row = sum((nx[i] - 1) * (ny[i] - 1))
        M_col = sum((nt - 1) * (ny[i] - 1))
    \gamma_w size: sum((nt - 1) * (ny[i] - 1))
    Parameters
    ----------
    mtx[system_size, system_size] : numpy array
        Final fully assembled AIC matrix that is used to solve for the
        circulations.
    rhs[system_size] : numpy array
        Right-hand side of the AIC linear system, constructed from the
        freestream velocities and panel normals.
    Returns
    -------
    circulations[system_size] : numpy array
        The vortex ring circulations obtained by solving the AIC linear system.
    """
    def initialize(self):
        self.parameters.declare('method',
                                values=['fw_euler', 'bk_euler'],
                                default='fw_euler')
        self.parameters.declare('nt', types=int)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('bd_vortex_shapes', types=list)
        self.parameters.declare('n', default=1)
        self.parameters.declare('delta_t')
        # pass

    def define(self):
        surface_names = self.parameters['surface_names']
        bd_vortex_shapes = self.parameters['bd_vortex_shapes']
        surface_shapes = bd_vortex_shapes
        method = self.parameters['method']
        nt = self.parameters['nt']
        n = self.parameters['n']
        delta_t = self.parameters['delta_t']

        bd_coll_pts_shapes = [
            tuple(map(lambda i, j: i - j, item, (1, 1, 0)))
            for item in bd_vortex_shapes
        ]

        # print('bd_coll_pts_shapes', bd_coll_pts_shapes)

        bd_vtx_coords_names = [x + '_bd_vtx_coords' for x in surface_names]
        coll_pts_coords_names = [x + '_coll_pts_coords' for x in surface_names]

        bd_vtx_normals = [x + '_bd_vtx_normals' for x in surface_names]
        # aic_bd_proj_names = [x + '_aic_bd_proj' for x in surface_names]
        wake_vortex_pts_shapes = [
            tuple((nt, item[1], item[2])) for item in bd_vortex_shapes
        ]
        for i in range(len(bd_vortex_shapes)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]

        model = Model()
        '''1. add the rhs'''
        model.add(
            RHS(
                nt=nt,
                surface_names=surface_names,
                bd_vortex_shapes=bd_vortex_shapes,
                delta_t=delta_t,
            ), 'RHS_group')

        nt = self.parameters['nt']
        '''2. compute A_mtx'''
        m = AssembleAic(
            bd_coll_pts_names=coll_pts_coords_names,
            wake_vortex_pts_names=bd_vtx_coords_names,
            bd_coll_pts_shapes=bd_coll_pts_shapes,
            wake_vortex_pts_shapes=bd_vortex_shapes,
            full_aic_name='aic_bd',
            delta_t=delta_t,  # one line of wake vortex for fix wake
        )
        model.add(m, name='AssembleAic_bd')
        '''3. project the aic on to the bd_vertices'''
        aic_shape_row = aic_shape_col = 0
        for i in range(len(bd_coll_pts_shapes)):
            aic_shape_row += (bd_coll_pts_shapes[i][0] *
                              bd_coll_pts_shapes[i][1])
            aic_shape_col += ((bd_coll_pts_shapes[i][0]) *
                              (bd_coll_pts_shapes[i][1]))
        aic_bd_proj_name = 'aic_bd_proj'

        m = Projection(
            input_vel_names=['aic_bd'],
            normal_names=bd_vtx_normals,
            output_vel_names=aic_bd_proj_name,  # this is b
            input_vel_shapes=[(aic_shape_row, aic_shape_col, 3)
                              ],  #rotatonal_vel_shapes
            normal_shapes=bd_coll_pts_shapes,
        )
        model.add(m, name='Projection_aic_bd')
        self.add(model, 'prepossing_before_Solve')
        '''3. solve'''
        model = Model()
        M_shape_row = M_shape_col = 0

        for i in range(len(bd_coll_pts_shapes)):
            M_shape_row += (bd_coll_pts_shapes[i][0] *
                            bd_coll_pts_shapes[i][1])
            M_shape_col += ((wake_vortex_pts_shapes[i][0] - 1) *
                            (wake_vortex_pts_shapes[i][1] - 1))
        M_shape = (M_shape_row, M_shape_col)

        M = model.declare_variable('M', shape=M_shape)
        # TODO: fix this for mls
        gamma_b_shape = sum((i[0] - 1) * (i[1] - 1) for i in bd_vortex_shapes)

        aic_bd_proj_shape = (gamma_b_shape, ) + (gamma_b_shape, )

        aic_bd_proj = model.declare_variable(aic_bd_proj_name,
                                             shape=(aic_bd_proj_shape))
        gamma_b = model.declare_variable('gamma_b', shape=(gamma_b_shape))
        b = model.declare_variable('b', shape=(gamma_b_shape, ))

        model_concatenate_gamma_w = Model()
        sum_ny = sum((i[1] - 1) for i in bd_vortex_shapes)
        gamma_w = model_concatenate_gamma_w.create_output('gamma_w',
                                                          shape=(n, nt - 1,
                                                                 sum_ny))
        start = 0

        gamma_b = self.declare_variable('gamma_b', shape=gamma_b_shape)

        self.add(SeperateGammab(surface_names=surface_names,
                                surface_shapes=surface_shapes),
                 name='seperate_gamma_b_1')

        for i in range(len(bd_vortex_shapes)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            delta = ny - 1
            surface_gamma_w_name = surface_names[i] + '_gamma_w'
            surface_gamma_b_name = surface_names[i] + '_gamma_b'
            print(surface_gamma_b_name, 'surface_gamma_b_name')

            surface_gamma_b = model_concatenate_gamma_w.declare_variable(
                surface_gamma_b_name, shape=((nx - 1) * (ny - 1), ))
            print('surface_gamma_b', surface_gamma_b.shape)
            print('surface_gamma_b',
                  surface_gamma_b[(nx - 2) * (ny - 1):].shape)
            print('surface_gamma_b',
                  (nt - 1, surface_gamma_b[(nx - 2) * (ny - 1):].shape[0]))

            surface_gamma_w = csdl.expand(
                surface_gamma_b[(nx - 2) * (ny - 1):],
                (nt - 1, surface_gamma_b[(nx - 2) * (ny - 1):].shape[0]),
                indices='i->ji')
            gamma_w[:, :, start:start + delta] = csdl.reshape(
                surface_gamma_w,
                (n, surface_gamma_w.shape[0], surface_gamma_w.shape[1]))
            start += delta
        self.add(model_concatenate_gamma_w, name='concatenate_gamma_w')
        gamma_w_flatten = csdl.reshape(gamma_w,
                                       new_shape=(gamma_w.shape[1] *
                                                  gamma_w.shape[2], ))

        # if method == 'bk_euler':
        #     gamma_w = csdl.expand(gamma_b[(nx - 2) * (ny - 1):],
        #                           (nt - 1, gamma_b[(nx - 2) *
        #                                            (ny - 1):].shape[0]),
        #                           indices='i->ji')
        #     gamma_w_flatten = csdl.reshape(gamma_w,
        #                                    new_shape=(gamma_w.shape[0] *
        #                                               gamma_w.shape[1], ))

        # elif method == 'fw_euler':
        #     gamma_w = model.declare_variable('gamma_w',
        #                                      shape=(n, nt - 1, sum_ny))

        #     gamma_w_reshaped = csdl.reshape(gamma_w,
        #                                     new_shape=(nt - 1, sum_ny))

        #     gamma_w_flatten = csdl.reshape(
        #         gamma_w_reshaped,
        #         new_shape=(gamma_w_reshaped.shape[0] *
        #                    gamma_w_reshaped.shape[1], ))
        # print('shape---------------------',
        #       csdl.einsum(aic_bd_proj, gamma_b, subscripts='ij,j->i').shape)
        # print('M shape', M.shape, gamma_w_flatten.shape)
        # print(csdl.einsum(M, gamma_w_flatten, subscripts='ij,j->i').shape)
        # print(b.shape)
        y = csdl.einsum(aic_bd_proj, gamma_b,subscripts='ij,j->i') +\
                csdl.einsum(M, gamma_w_flatten, subscripts='ij,j->i')+b

        model.register_output('y', y)

        solve = self.create_implicit_operation(model)
        solve.declare_state('gamma_b', residual='y')
        solve.nonlinear_solver = NewtonSolver(
            solve_subsystems=False,
            maxiter=2,
            iprint=False,
        )
        solve.linear_solver = ScipyKrylov()

        M = self.declare_variable('M', shape=M_shape)
        # print('MMMMM', M.shape)
        aic_bd_proj = self.declare_variable(aic_bd_proj_name,
                                            shape=(aic_shape_row,
                                                   aic_shape_col))
        # print('nt----------', nt)
        gamma_w = self.declare_variable(
            'gamma_w',
            shape=(1, nt - 1, gamma_b[(nx - 2) * (ny - 1):].shape[0]))
        # print('gamma_w----------', gamma_w.shape)

        gamma_b = solve(M, aic_bd_proj)


if __name__ == "__main__":

    sim = Simulator(
        SolveMatrix(nt=3, surface_names=['wing'],
                    bd_vortex_shapes=[(2, 2, 3)]))
    sim.run()
    sim.visualize_implementation()
    # print('aic is', sim['aic'])
    # print('v_ind is', sim['v_ind'].shape, sim['v_ind'])