from csdl import Model, ImplicitOperation, ScipyKrylov, NewtonSolver, NonlinearBlockGS
import numpy as np
from csdl_om import Simulator
import csdl

from UVLM_package.UVLM_system.solve_circulations.rhs_group_new import RHS
from UVLM_package.UVLM_system.solve_circulations.assemble_aic import AssembleAic
from UVLM_package.UVLM_system.solve_circulations.compute_normal_comp import ComputeNormal
from UVLM_package.UVLM_system.solve_circulations.projection_comp import Projection


class SolveMatrix(Model):
    """
    Solve the AIC linear system to obtain the vortex ring circulations.
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
        # self.parameters.declare('A_mtx_shape', types=tuple)
        self.parameters.declare('method',
                                values=['fw_euler', 'bk_euler'],
                                default='fw_euler')
        self.parameters.declare('nt', types=int)
        self.parameters.declare('bd_vortex_shapes', types=list)
        # pass

    def define(self):
        bd_vortex_shapes = self.parameters['bd_vortex_shapes']
        for i in range(len(bd_vortex_shapes)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
        # nx = 2
        # ny = 4
        # nt = 5

        method = self.parameters['method']
        nt = self.parameters['nt']
        model = Model()
        '''1. add the rhs'''
        model.add(RHS(nt=nt,bd_vortex_shapes=bd_vortex_shapes), 'RHS_group')

        nt = self.parameters['nt']
        '''2. compute A_mtx'''
        m = AssembleAic(
            bd_coll_pts_names=['coll_coords'],
            wake_vortex_pts_names=['bd_vortex_coords'],
            bd_coll_pts_shapes=[((nx - 1), (ny - 1), 3)],
            wake_vortex_pts_shapes=[(nx, ny, 3)],
            full_aic_name='aic_bd'  # one line of wake vortex for fix wake
        )
        model.add(m, name='AssembleAic_bd')
        '''3. project the aic on to the bd_vertices'''
        m = Projection(
            input_vel_names=['aic_bd'],
            normal_names=['bd_vtx_normals'],
            output_vel_names=['aic_bd_proj'],  # this is b
            input_vel_shapes=[((nx - 1) * (ny - 1), (nx - 1) * (ny - 1), 3)
                              ],  #rotatonal_vel_shapes
            normal_shapes=[((nx - 1), (ny - 1), 3)],
        )
        model.add(m, name='Projection_aic_bd')
        self.add(model, 'prepossing_before_Solve')
        '''3. solve'''
        model = Model()
        M_shape = ((nx - 1) * (ny - 1), (nt - 1) * (ny - 1))

        M = model.declare_variable('M', shape=M_shape)
        aic_bd_proj = model.declare_variable('aic_bd_proj',
                                             shape=((nx - 1) * (ny - 1),
                                                    (nx - 1) * (ny - 1)))
        gamma_b = model.declare_variable('gamma_b',
                                         shape=((nx - 1) * (ny - 1), ))
        b = model.declare_variable('b', shape=((nx - 1) * (ny - 1), ))

        if method == 'bk_euler':
            gamma_w = csdl.expand(gamma_b[(nx - 2) * (ny - 1):],
                                  (nt - 1, gamma_b[(nx - 2) *
                                                   (ny - 1):].shape[0]),
                                  indices='i->ji')
            gamma_w_flatten = csdl.reshape(gamma_w,
                                           new_shape=(gamma_w.shape[0] *
                                                      gamma_w.shape[1], ))

        elif method == 'fw_euler':
            gamma_w = model.declare_variable(
                'gamma_w',
                shape=(1, nt - 1, gamma_b[(nx - 2) * (ny - 1):].shape[0]))
            print('gamma_w----before------', gamma_w.shape)
            gamma_w_reshaped = csdl.reshape(
                gamma_w,
                new_shape=(nt - 1, gamma_b[(nx - 2) * (ny - 1):].shape[0]))
            # gamma_w = model.declare_variable('gamma_w',
            #                                  val=np.zeros((nt - 1, ny - 1)))
            gamma_w_flatten = csdl.reshape(
                gamma_w_reshaped,
                new_shape=(gamma_w_reshaped.shape[0] *
                           gamma_w_reshaped.shape[1], ))
            # print('**********shape****gamma_w_flatten', gamma_w_flatten.shape)

        y = csdl.einsum(aic_bd_proj, gamma_b,subscripts='ij,j->i') +\
                csdl.einsum(M, gamma_w_flatten, subscripts='ij,j->i')+b

        # y = csdl.einsum(aic_bd_proj, gamma_b,subscripts='ij,j->i') +\
        #         csdl.einsum(M, gamma_b[3:], subscripts='ij,j->i')-b

        model.register_output('y', y)

        solve = self.create_implicit_operation(model)
        solve.declare_state('gamma_b', residual='y')
        solve.nonlinear_solver = NewtonSolver(
            solve_subsystems=False,
            maxiter=10,
            iprint=False,
        )
        solve.linear_solver = ScipyKrylov()

        M = self.declare_variable('M', shape=M_shape)
        print('MMMMM', M.shape)
        aic_bd_proj = self.declare_variable('aic_bd_proj',
                                            shape=((nx - 1) * (ny - 1),
                                                   (nx - 1) * (ny - 1)))
        print('nt----------', nt)
        gamma_w = self.declare_variable(
            'gamma_w',
            shape=(1, nt - 1, gamma_b[(nx - 2) * (ny - 1):].shape[0]))
        print('gamma_w----------', gamma_w.shape)

        gamma_b = solve(M, aic_bd_proj, gamma_w)


if __name__ == "__main__":

    def generate_simple_mesh(nx, ny, nt=None):
        if nt == None:
            mesh = np.zeros((nx, ny, 3))
            mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))
            mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
            mesh[:, :, 2] = 0.
        else:
            mesh = np.zeros((nt, nx, ny, 3))
            for i in range(nt):
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
    model_1.add(SolveMatrix())

    sim = Simulator(model_1)
    sim.run()
    sim.visualize_implementation()
    # print('aic is', sim['aic'])
    # print('v_ind is', sim['v_ind'].shape, sim['v_ind'])