from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size
import random

from openaerostruct_csdl.utils.vector_algebra import add_ones_axis
from openaerostruct_csdl.utils.vector_algebra import compute_dot, compute_dot_deriv
from openaerostruct_csdl.utils.vector_algebra import compute_cross, compute_cross_deriv1, compute_cross_deriv2
from openaerostruct_csdl.utils.vector_algebra import compute_norm, compute_norm_deriv

tol = 1e-10


def _compute_finite_vortex(r1, r2):
    r1_norm = compute_norm(r1)
    r2_norm = compute_norm(r2)

    r1_x_r2 = compute_cross(r1, r2)
    r1_d_r2 = compute_dot(r1, r2)

    num = (1. / r1_norm + 1. / r2_norm) * r1_x_r2
    den = r1_norm * r2_norm + r1_d_r2

    result = np.divide(num,
                       den * 4 * np.pi,
                       out=np.zeros_like(num),
                       where=np.abs(den) > tol)

    return result


def _compute_finite_vortex_deriv1(r1, r2, r1_deriv):
    r1_norm = add_ones_axis(compute_norm(r1))
    r2_norm = add_ones_axis(compute_norm(r2))
    r1_norm_deriv = compute_norm_deriv(r1, r1_deriv)

    r1_x_r2 = add_ones_axis(compute_cross(r1, r2))
    r1_d_r2 = add_ones_axis(compute_dot(r1, r2))
    r1_x_r2_deriv = compute_cross_deriv1(r1_deriv, r2)
    r1_d_r2_deriv = compute_dot_deriv(r2, r1_deriv)

    num = (1. / r1_norm + 1. / r2_norm) * r1_x_r2
    num_deriv = (-r1_norm_deriv / r1_norm ** 2) * r1_x_r2 \
        + (1. / r1_norm + 1. / r2_norm) * r1_x_r2_deriv

    den = r1_norm * r2_norm + r1_d_r2
    den_deriv = r1_norm_deriv * r2_norm + r1_d_r2_deriv

    result = np.divide(num_deriv * den - num * den_deriv,
                       den**2 * 4 * np.pi,
                       out=np.zeros_like(num),
                       where=np.abs(den) > tol)

    return result


def _compute_finite_vortex_deriv2(r1, r2, r2_deriv):
    r1_norm = add_ones_axis(compute_norm(r1))
    r2_norm = add_ones_axis(compute_norm(r2))
    r2_norm_deriv = compute_norm_deriv(r2, r2_deriv)

    r1_x_r2 = add_ones_axis(compute_cross(r1, r2))
    r1_d_r2 = add_ones_axis(compute_dot(r1, r2))
    r1_x_r2_deriv = compute_cross_deriv2(r1, r2_deriv)
    r1_d_r2_deriv = compute_dot_deriv(r1, r2_deriv)

    num = (1. / r1_norm + 1. / r2_norm) * r1_x_r2
    num_deriv = (-r2_norm_deriv / r2_norm ** 2) * r1_x_r2 \
        + (1. / r1_norm + 1. / r2_norm) * r1_x_r2_deriv

    den = r1_norm * r2_norm + r1_d_r2
    den_deriv = r1_norm * r2_norm_deriv + r1_d_r2_deriv

    result = np.divide(num_deriv * den - num * den_deriv,
                       den**2 * 4 * np.pi,
                       out=np.zeros_like(num),
                       where=np.abs(den) > tol)

    return result


class BiotSvart(csdl.CustomExplicitOperation):
    """
    Compute AIC.

    parameters
    ----------
    eval_pts[nx, ny, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface.
    vortex_coords[nx, ny, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface.

    Returns
    -------
    AIC[nx-1, ny, 3] : numpy array
        Aerodynamic influence coeffients (can be interprete as induced
        velocities given circulations=1)
    note: there should not be nt-th dimension in BiotSvart as for now
    """
    def initialize(self):
        self.parameters.declare('eval_pt_names', types=list)
        self.parameters.declare('vortex_coords_names', types=list)

        self.parameters.declare('eval_pt_shapes', types=list)
        self.parameters.declare('vortex_coords_shapes', types=list)

        self.parameters.declare('output_names', types=list)

    def define(self):
        eval_pt_names = self.parameters['eval_pt_names']
        vortex_coords_names = self.parameters['vortex_coords_names']
        eval_pt_shapes = self.parameters['eval_pt_shapes']
        vortex_coords_shapes = self.parameters['vortex_coords_shapes']
        output_names = self.parameters['output_names']

        for i in range(len(eval_pt_names)):
            # input_names
            eval_pt_name = eval_pt_names[i]
            vortex_coords_name = vortex_coords_names[i]

            # output_name
            output_name = output_names[i]

            # input_shapes
            eval_pt_shape = eval_pt_shapes[i]
            vortex_coords_shape = vortex_coords_shapes[i]

            self.add_input(eval_pt_name, shape=eval_pt_shape)
            self.add_input(vortex_coords_name, shape=vortex_coords_shape)

            self.add_output(output_name, shape=(6, 6, 3))
        self.declare_derivatives('*', '*')

    def compute(self, inputs, outputs):

        eval_pt_names = self.parameters['eval_pt_names']
        vortex_coords_names = self.parameters['vortex_coords_names']
        eval_pt_shapes = self.parameters['eval_pt_shapes']
        vortex_coords_shapes = self.parameters['vortex_coords_shapes']
        output_names = self.parameters['output_names']

        for i in range(len(eval_pt_names)):

            # input_names
            eval_pt_name = eval_pt_names[i]
            vortex_coords_name = vortex_coords_names[i]

            # output_name
            output_name = output_names[i]

            # input_shapes
            eval_pt_shape = eval_pt_shapes[i]
            vortex_coords_shape = vortex_coords_shapes[i]

            # inputs
            eval_pts = inputs[eval_pt_name]
            vortex_coords = inputs[vortex_coords_name]

            # define panel points
            #                  C -----> D
            # ---v_inf-(x)-->  ^        |
            #                  |        v
            #                  B <----- A
            A = vortex_coords[1:, :vortex_coords_shape[1] - 1, :]
            B = vortex_coords[:vortex_coords_shape[0] -
                              1, :vortex_coords_shape[1] - 1, :]
            C = vortex_coords[:vortex_coords_shape[0] - 1, 1:, :]
            D = vortex_coords[1:, 1:, :]
            # print('Ashape', A.shape)

            v_ab = self._induced_vel_line(eval_pts, A, B)
            v_bc = self._induced_vel_line(eval_pts, B, C)
            v_cd = self._induced_vel_line(eval_pts, C, D)
            v_da = self._induced_vel_line(eval_pts, D, A)
            outputs[output_name] = v_ab + v_bc + v_cd + v_da

    def _induced_vel_line(self, eval_pts, p_1, p_2):
        # 1 -> 2 eval_pts(num_pts_x,num_pts_y, 3)
        # v_induced_line shape=(num_panel_x*num_panel_y, num_panel_x, num_panel_y, 3)
        num_repeat_eval = p_1.shape[0] * p_1.shape[1]
        num_repeat_p = eval_pts.shape[0] * eval_pts.shape[1]

        # print('########################################', num_repeat_eval,
        #   num_repeat_p)

        # ones_num_repeat_p_var = self.declare_variable(
        #     'ones_num_repeat_p', val=np.ones(num_repeat_p))
        '''e1_____________________'''

        eval_pts_expand = np.einsum('ij, k -> ikj', eval_pts.reshape(-1, 3),
                                    np.ones((num_repeat_eval))).reshape(-1, 3)
        p_1_expand = np.einsum('ij, k -> kij', p_1.reshape(-1, 3),
                               np.ones((num_repeat_p))).reshape(-1, 3)
        p_2_expand = np.einsum('ij, k -> kij', p_2.reshape(-1, 3),
                               np.ones((num_repeat_p))).reshape(-1, 3)

        r1 = eval_pts_expand - p_1_expand
        r2 = eval_pts_expand - p_2_expand
        r0 = p_2_expand - p_1_expand

        r1_norm = compute_norm(r1)
        r2_norm = compute_norm(r2)

        r1_x_r2 = compute_cross(r1, r2)
        r1_d_r2 = compute_dot(r1, r2)

        num = (1. / r1_norm + 1. / r2_norm) * r1_x_r2
        den = r1_norm * r2_norm + r1_d_r2

        result = np.divide(num,
                           den * 4 * np.pi,
                           out=np.zeros_like(num),
                           where=np.abs(den) > tol)

        # TODO: fix this tomorrow morning
        # v_induced_line[np.isnan(v_induced_line)] = 0

        return result


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

    eval_pt_names = ['col']
    vortex_coords_names = ['vor']
    eval_pt_shapes = [(2, 3, 3)]
    # eval_pt_shapes = [(3, 4, 3)]

    vortex_coords_shapes = [(3, 4, 3)]

    output_names = ['aic']

    model_1 = Model()
    vor_val = generate_simple_mesh(3, 4)
    col_val = 0.25 * (vor_val[:-1, :-1, :] + vor_val[:-1, 1:, :] +
                      vor_val[1:, :-1, :] + vor_val[1:, 1:, :])

    # col_val = generate_simple_mesh(3, 4)

    vor = model_1.create_input('vor', val=vor_val)
    col = model_1.create_input('col', val=col_val)
    model_1.add(BiotSvart(eval_pt_names=eval_pt_names,
                          vortex_coords_names=vortex_coords_names,
                          eval_pt_shapes=eval_pt_shapes,
                          vortex_coords_shapes=vortex_coords_shapes,
                          output_names=output_names),
                name='BiotSvart_group')
    sim = Simulator(model_1)

    print(sim['vor'])
    print(sim[output_names[0]])
    # sim.visualize_implementation()
    # sim.run()

# from csdl_om import Simulator
# import csdl
# from csdl import Model
# import numpy as np

# val = np.array([
#     [1., 2., 3.],
#     [4., 5., 6.],
# ])
# array = Model().declare_variable('array', val=val)
# expanded_array = csdl.expand(array, (2, 2, 3), 'ij->kij')
# self.register_output('expanded_array', expanded_array)
