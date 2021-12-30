from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size
import random
# how to declare.options[]? Since I need 'if' statement
# how to set values of the variables outside the class (Indep_var comp?)
# why do we need a create input-how to set its value outside the class
# How to use the same input
# name of the comps, can it be something meaningful
# size differnet than actual python code
# what is the rule for registering outputs
# cannot reshape chords
# prjected vs wetted s_ref?
# line 153 why cannot put 0.5 outside the ()


class BiotSvart(Model):
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
        # print('**************')

        for i in range(len(eval_pt_names)):
            # print(i)
            # print('************')

            # input_names
            eval_pt_name = eval_pt_names[i]
            vortex_coords_name = vortex_coords_names[i]

            # output_name
            output_name = output_names[i]

            # input_shapes
            eval_pt_shape = eval_pt_shapes[i]
            vortex_coords_shape = vortex_coords_shapes[i]

            # declare_inputs
            # print('evalshape', eval_pt_shape)
            eval_pts = self.declare_variable(eval_pt_name, shape=eval_pt_shape)
            vortex_coords = self.declare_variable(vortex_coords_name,
                                                  shape=vortex_coords_shape)

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
            AIC = v_ab + v_bc + v_cd + v_da
            # print('AIC', AIC.shape)
            self.register_output(output_name, AIC)
            # print('AIC', AIC.shape)

    def _induced_vel_line(self, eval_pts, p_1, p_2):
        # print('_induced_vel_line')
        # 1 -> 2 eval_pts(num_pts_x,num_pts_y, 3)
        # v_induced_line shape=(num_panel_x*num_panel_y, num_panel_x, num_panel_y, 3)
        num_repeat_eval = p_1.shape[0] * p_1.shape[1]
        num_repeat_p = eval_pts.shape[0] * eval_pts.shape[1]

        ones_num_repeat_eval_var = self.declare_variable(
            'ones_num_repeat_eval', val=np.ones(num_repeat_eval))
        # print('########################################', num_repeat_eval,
        #   num_repeat_p)

        # ones_num_repeat_p_var = self.declare_variable(
        #     'ones_num_repeat_p', val=np.ones(num_repeat_p))
        '''e1_____________________'''

        eval_pts_expand = csdl.reshape(
            csdl.expand(
                csdl.reshape(eval_pts,
                             new_shape=((eval_pts.shape[0] *
                                         eval_pts.shape[1]), 3)),
                (eval_pts.shape[0] * eval_pts.shape[1], num_repeat_eval, 3),
                'ij->ikj',
            ),
            new_shape=(eval_pts.shape[0] * eval_pts.shape[1] * num_repeat_eval,
                       3))

        p_1_expand = csdl.reshape(csdl.expand(
            csdl.reshape(p_1, new_shape=((p_1.shape[0] * p_1.shape[1]), 3)),
            (num_repeat_p, p_1.shape[0] * p_1.shape[1], 3), 'ij->kij'),
                                  new_shape=(p_1.shape[0] * p_1.shape[1] *
                                             num_repeat_p, 3))

        p_2_expand = csdl.reshape(csdl.expand(
            csdl.reshape(p_2, new_shape=((p_2.shape[0] * p_2.shape[1]), 3)),
            (num_repeat_p, p_2.shape[0] * p_2.shape[1], 3), 'ij->kij'),
                                  new_shape=(p_2.shape[0] * p_2.shape[1] *
                                             num_repeat_p, 3))

        r1 = eval_pts_expand - p_1_expand
        r2 = eval_pts_expand - p_2_expand
        r0 = p_2_expand - p_1_expand
        # book pg269
        # step 1 calculate r1_x_r2,r1_x_r2_norm_sq
        ones_3_val = self.declare_variable(
            'ones_3',
            val=np.ones(3))  # TODO: substitude the einsum with expand
        # ones_shape_val = self.declare_variable('ones_s', val=np.ones(123))
        r1_x_r2 = csdl.cross(r1, r2, axis=1)
        # print(r1_x_r2.shape)

        r1_x_r2_norm_sq = csdl.einsum(
            csdl.sum(r1_x_r2**2, axes=(1, )),
            ones_3_val,
            subscripts='i,k->ik',
            # subscripts='...,k->...k',
        )

        # self.register_output('e4_out' + str(random.randint(0, 1000)),
        #                      r1_x_r2_norm_sq)

        # step 2 r1_norm, r2_norm
        r1_norm = csdl.einsum(
            csdl.sum(r1**2, axes=(1, ))**0.5,
            ones_3_val,
            subscripts='i,k->ik',
        )
        r2_norm = csdl.einsum(
            csdl.sum(r2**2, axes=(1, ))**0.5,
            ones_3_val,
            subscripts='i,k->ik',
        )
        # print('array1 shapes', r1_x_r2.shape, r1_x_r2_norm_sq.shape)
        array1 = 1 / (np.pi * 4) * r1_x_r2 / r1_x_r2_norm_sq

        array2 = ((csdl.einsum(
            (r1 / r1_norm - r2 / r2_norm),
            r0,
            subscripts='ij,ij->i',
        )))
        # print('array12 shapes', array1.shape, array2.shape)
        v_induced_line = array1 * csdl.expand(array2, array1.shape, 'i->ij')
        # print('************')
        # TODO: fix this tomorrow morning
        # v_induced_line[np.isnan(v_induced_line)] = 0

        return v_induced_line


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
    vortex_coords_shapes = [(3, 4, 3)]
    output_names = ['aic']

    model_1 = Model()
    vor_val = generate_simple_mesh(3, 4)
    col_val = 0.25 * (vor_val[:-1, :-1, :] + vor_val[:-1, 1:, :] +
                      vor_val[1:, :-1, :] + vor_val[1:, 1:, :])

    vor = model_1.create_input('vor', val=vor_val)
    col = model_1.create_input('col', val=col_val)
    model_1.add(BiotSvart(eval_pt_names=eval_pt_names,
                          vortex_coords_names=vortex_coords_names,
                          eval_pt_shapes=eval_pt_shapes,
                          vortex_coords_shapes=vortex_coords_shapes,
                          output_names=output_names),
                name='BiotSvart_group')
    sim = Simulator(model_1)
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
