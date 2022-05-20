from csdl import Model, ImplicitOperation, ScipyKrylov, NewtonSolver, NonlinearBlockGS
import numpy as np
from csdl_om import Simulator
import csdl

from VLM_package.VLM_system.solve_circulations.rhs_group import RHS
from VLM_package.VLM_system.solve_circulations.assemble_aic import AssembleAic
from VLM_package.VLM_system.solve_circulations.projection_comp import Projection


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
                                default='bk_euler')
        self.parameters.declare('nt', types=int)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('bd_vortex_shapes', types=list)
        self.parameters.declare('n', default=1)
        self.parameters.declare('delta_t')
        # pass

    def define(self):
        surface_names = self.parameters['surface_names']
        bd_vortex_shapes = self.parameters['bd_vortex_shapes']
        num_nodes = bd_vortex_shapes[0][0]

        method = self.parameters['method']
        nt = self.parameters['nt']
        n = self.parameters['n']
        delta_t = self.parameters['delta_t']

        bd_coll_pts_shapes = [
            tuple(map(lambda i, j: i - j, item, (0, 1, 1, 0)))
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
        self.add(model, 'prepossing_before_Solve')
        '''3. solve'''
        model = Model()
        M_shape_row = M_shape_col = 0

        for i in range(len(bd_coll_pts_shapes)):
            M_shape_row += (bd_coll_pts_shapes[i][1] *
                            bd_coll_pts_shapes[i][2])
            M_shape_col += ((wake_vortex_pts_shapes[i][1] - 1) *
                            (wake_vortex_pts_shapes[i][2] - 1))

        M_shape = (M_shape_row, M_shape_col)
        M = model.declare_variable('M', shape=M_shape)
        M_reshaped = model.declare_variable('M_reshaped',
                                            shape=(num_nodes, M_shape_row,
                                                   M_shape_row))

        gamma_b_shape = sum((i[1] - 1) * (i[2] - 1) for i in bd_vortex_shapes)
        # this is gamma_b_shape per time step
        # print('solve_group gamma_b_shape', gamma_b_shape)

        aic_bd_proj_shape = \
            (num_nodes, ) + (gamma_b_shape, ) + (gamma_b_shape, )
        '''declare terms in the equations'''
        MTX = model.declare_variable('MTX', shape=(aic_bd_proj_shape))
        # print('solve_group before implicit MTX shape', MTX.shape)
        # print('solve_group before implicit M_reshaped shape', M_reshaped.shape)
        gamma_b = model.declare_variable('gamma_b',
                                         shape=(num_nodes, gamma_b_shape))
        b = model.declare_variable('b', shape=(num_nodes, gamma_b_shape))

        # if method == 'bk_euler':
        #     gamma_w = model.create_output('gamma_w', shape=(n, nt - 1, sum_ny))
        #     start = start_b = 0
        #     for i in range(len(surface_names)):
        #         nx = bd_vortex_shapes[i][0]
        #         ny = bd_vortex_shapes[i][1]
        #         delta = ny - 1
        #         delta_b = (nx - 1) * (ny - 1)
        #         val = np.zeros((n, nt - 1, ny - 1))
        #         surface_name = surface_names[i]

        #         surface_gamma_b = gamma_b[start:start + delta_b]
        #         surface_gamma_w_name = surface_names[i] + '_gamma_w'
        #         surface_gamma_w = csdl.expand(
        #             surface_gamma_b[(nx - 2) * (ny - 1):], (nt - 1, ny - 1),
        #             'i->ji')
        #         model.register_output(surface_gamma_w_name, surface_gamma_w)
        #         gamma_w[:, :, start:start + delta] = csdl.reshape(
        #             surface_gamma_w, (1, nt - 1, ny - 1))
        #         start += delta
        #         start_b += delta_b

        #     gamma_w_flatten = csdl.reshape(gamma_w,
        #                                    new_shape=(gamma_w.shape[1] *
        #                                               gamma_w.shape[2], ))

        # sum_ny = sum((i[1] - 1) for i in bd_vortex_shapes)
        # gamma_b[(nx - 2) * (ny - 1):]
        # y = csdl.einsum(aic_bd_proj, gamma_b,
        #                 subscripts='ij,j->i') + csdl.einsum(
        #                     M, gamma_w_flatten, subscripts='ij,j->i') + b

        y = csdl.einsum(MTX, gamma_b, subscripts='kij,kj->ki') + b

        model.register_output('y', y)

        solve = self.create_implicit_operation(model)
        solve.declare_state('gamma_b', residual='y')
        solve.nonlinear_solver = NewtonSolver(
            solve_subsystems=False,
            maxiter=5,
            iprint=True,
        )
        solve.linear_solver = ScipyKrylov()

        # aic_bd_proj = self.declare_variable(aic_bd_proj_name,
        #                                     shape=(aic_shape_row,
        #                                            aic_shape_col))
        MTX = self.declare_variable('MTX',
                                    shape=(num_nodes, M_shape_row,
                                           M_shape_row))
        # M = self.declare_variable('M', shape=M_shape)
        b = self.declare_variable('b', shape=(num_nodes, gamma_b_shape))
        # print('solve_group after implicit M_shape_row', M_shape_row)
        # print('solve_group after implicit MTX shape', MTX.shape)

        gamma_b = solve(MTX, b)
        # model.visualize_sparsity(recursive=True)


if __name__ == "__main__":

    sim = Simulator(
        SolveMatrix(nt=3, surface_names=['wing'],
                    bd_vortex_shapes=[(2, 2, 3)]))
    sim.run()
    sim.visualize_implementation()
    # print('aic is', sim['aic'])
    # print('v_ind is', sim['v_ind'].shape, sim['v_ind'])