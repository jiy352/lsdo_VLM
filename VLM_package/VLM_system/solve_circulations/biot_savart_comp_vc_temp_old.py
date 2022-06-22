from random import randint
from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np


class BiotSavartComp(Model):
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
    note: there should not be n_wake_pts_chord-th dimension in BiotSavartComp as for now
    """
    def initialize(self):
        self.parameters.declare('eval_pt_names', types=list)
        self.parameters.declare('vortex_coords_names', types=list)

        self.parameters.declare('eval_pt_shapes', types=list)
        self.parameters.declare('vortex_coords_shapes', types=list)

        self.parameters.declare('output_names', types=list)

        self.parameters.declare('vc', default=False)
        self.parameters.declare('n_wake_pts_chord', default=1)
        self.parameters.declare('delta_t')
        self.parameters.declare('eps', default=1e-10)

        self.parameters.declare('circulation_names', default=None)

    def define(self):
        eval_pt_names = self.parameters['eval_pt_names']
        vortex_coords_names = self.parameters['vortex_coords_names']
        eval_pt_shapes = self.parameters['eval_pt_shapes']
        vortex_coords_shapes = self.parameters['vortex_coords_shapes']
        output_names = self.parameters['output_names']
        circulation_names = self.parameters['circulation_names']

        for i in range(len(eval_pt_names)):

            # input_names
            eval_pt_name = eval_pt_names[i]
            vortex_coords_name = vortex_coords_names[i]
            if self.parameters['vc'] == True:
                if circulation_names != None:
                    circulation_name = circulation_names[i]
                else:
                    circulation_name = None
            else:
                circulation_name = None

            # output_name
            output_name = output_names[i]

            # input_shapes
            eval_pt_shape = eval_pt_shapes[i]
            vortex_coords_shape = vortex_coords_shapes[i]

            # declare_inputs
            eval_pts = self.declare_variable(eval_pt_name, shape=eval_pt_shape)
            vortex_coords = self.declare_variable(vortex_coords_name,
                                                  shape=vortex_coords_shape)

            # define panel points
            #                  C -----> D
            # ---v_inf-(x)-->  ^        |
            #                  |        v
            #                  B <----- A
            # A = vortex_coords[:,1:, :vortex_coords_shape[1] - 1, :]
            # B = vortex_coords[:,:vortex_coords_shape[0] -
            #                   1, :vortex_coords_shape[1] - 1, :]
            # C = vortex_coords[:,:vortex_coords_shape[0] - 1, 1:, :]
            # D = vortex_coords[:,1:, 1:, :]

            # openaerostruct
            C = vortex_coords[:, 1:, :vortex_coords_shape[2] - 1, :]
            B = vortex_coords[:, :vortex_coords_shape[1] -
                              1, :vortex_coords_shape[2] - 1, :]
            A = vortex_coords[:, :vortex_coords_shape[1] - 1, 1:, :]
            D = vortex_coords[:, 1:, 1:, :]
            # print('BScomp l91 C shape', C.shape)
            # print('BScomp l92 B shape', B.shape)
            # print('BScomp l93 A shape', A.shape)
            # print('BScomp l94 D shape', D.shape)

            v_ab = self._induced_vel_line(eval_pts, A, B, vortex_coords_shape,
                                          circulation_name)
            v_bc = self._induced_vel_line(eval_pts, B, C, vortex_coords_shape,
                                          circulation_name)
            v_cd = self._induced_vel_line(eval_pts, C, D, vortex_coords_shape,
                                          circulation_name)
            v_da = self._induced_vel_line(eval_pts, D, A, vortex_coords_shape,
                                          circulation_name)
            AIC = v_ab + v_bc + v_cd + v_da
            self.register_output(output_name, AIC)

    def _induced_vel_line(self, eval_pts, p_1, p_2, vortex_coords_shape,
                          circulation_name):

        vc = self.parameters['vc']
        n_wake_pts_chord = self.parameters['n_wake_pts_chord']
        delta_t = self.parameters['delta_t']
        num_nodes = eval_pts.shape[0]

        # 1 -> 2 eval_pts(num_pts_x,num_pts_y, 3)
        # v_induced_line shape=(num_panel_x*num_panel_y, num_panel_x, num_panel_y, 3)
        num_repeat_eval = p_1.shape[1] * p_1.shape[2]
        num_repeat_p = eval_pts.shape[1] * eval_pts.shape[2]

        eval_pts_expand = csdl.reshape(
            csdl.expand(
                csdl.reshape(eval_pts,
                             new_shape=(num_nodes, (eval_pts.shape[1] *
                                                    eval_pts.shape[2]), 3)),
                (num_nodes, eval_pts.shape[1] * eval_pts.shape[2],
                 num_repeat_eval, 3),
                'lij->likj',
            ),
            new_shape=(num_nodes,
                       eval_pts.shape[1] * eval_pts.shape[2] * num_repeat_eval,
                       3))
        # self.register_output('eval_pts_expand' + str(randint(0, 1000)),
        #                      eval_pts_expand)


        p_1_expand = csdl.reshape(\
            csdl.expand(
                csdl.reshape(p_1,
                         new_shape=(num_nodes, (p_1.shape[1] * p_1.shape[2]),
                                    3)),
            (num_nodes, num_repeat_p, p_1.shape[1] * p_1.shape[2], 3),
            'lij->lkij'),
                        new_shape=(num_nodes,
                                    p_1.shape[1] *p_1.shape[2] * num_repeat_p,
                                    3))

        p_2_expand = csdl.reshape(\
            csdl.expand(
                csdl.reshape(p_2,
                            new_shape=(num_nodes, (p_2.shape[1] * p_2.shape[2]),
                                        3)),
                (num_nodes, num_repeat_p, p_2.shape[1] * p_2.shape[2], 3),
                'lij->lkij'),
                            new_shape=(num_nodes,
                                        p_2.shape[1] *p_2.shape[2] * num_repeat_p,
                                        3))
        # print('BScomp l154 eval_pts_expand shape', eval_pts_expand.shape)
        # print('BScomp l155 p_1_expand shape', p_1_expand.shape)
        # print('BScomp l156 p_2_expand shape', p_2_expand.shape)
        # print('BScomp l156 p_1 shape', p_1.shape)
        # print('BScomp l156 p_2 shape', p_2.shape)

        r1 = eval_pts_expand - p_1_expand
        r2 = eval_pts_expand - p_2_expand
        r0 = p_2_expand - p_1_expand

        r1_x_r2_norm_shape = num_repeat_eval * num_repeat_p

        # book pg269
        # step 1 calculate r1_x_r2,r1_x_r2_norm_sq

        # ones_shape_val = self.declare_variable('ones_s', val=np.ones(123))
        r1_x_r2 = csdl.cross(r1, r2, axis=2)

        r1_x_r2_norm_sq = csdl.expand(csdl.sum(r1_x_r2**2, axes=(2, )),
                                      shape=(num_nodes, r1_x_r2_norm_shape, 3),
                                      indices=('ki->kij'))

        # step 2 r1_norm, r2_norm
        r1_norm = csdl.expand(csdl.sum(r1**2 + self.parameters['eps'],
                                       axes=(2, ))**0.5,
                              shape=(num_nodes, r1_x_r2_norm_shape, 3),
                              indices=('ki->kij'))

        r2_norm = csdl.expand(csdl.sum(r2**2 + self.parameters['eps'],
                                       axes=(2, ))**0.5,
                              shape=(num_nodes, r1_x_r2_norm_shape, 3),
                              indices=('ki->kij'))

        array1 = 1 / (np.pi * 4) * r1_x_r2

        if vc == True:
            a_l = 1.25643
            kinematic_viscocity = 1.48 * 1e-5
            a_1 = 0.1
            nx = vortex_coords_shape[1]
            ny = vortex_coords_shape[2]
            # TODO fix here
            circulations = self.declare_variable(circulation_name,
                                                 shape=(num_nodes,
                                                        (nx - 1) * (ny - 1)))
            # print('shape-----------------', circulations.shape)

            time_current = 1  # TODO fix this time input
            # print(time_current, 'time_current')

            # print('shape-----------------', circulations.shape[1:])

            sigma = 1 + a_1 * csdl.reshape(
                circulations, new_shape=(num_nodes, (nx - 1),
                                         (ny - 1))) / kinematic_viscocity

            r_c = (4 * a_l * kinematic_viscocity * sigma * time_current +
                   self.parameters['eps']
                   )**0.5  # size = (n_wake_pts_chord-1, ny-1)

            # r2_r1_norm_sq = csdl.sum((r2 - r1)**2, axes=(1, ))

            rc_sq = r_c**2

            rc_sq_reshaped = csdl.reshape(rc_sq,
                                          new_shape=(
                                              num_nodes,
                                              rc_sq.shape[1] * rc_sq.shape[2],
                                          ))
            # rc_sq_expand = csdl.reshape(
            #     csdl.expand(rc_sq_reshaped,
            #                 (num_repeat_p, rc_sq_reshaped.shape[0]), 'i->ji'),
            #     new_shape=(rc_sq.shape[0] * rc_sq.shape[1] * num_repeat_p))

            # print('print shapes')
            # print('num_repeat_eval', num_repeat_eval)
            # print('num_repeat_p', num_repeat_p)
            # print('circulations', circulations.shape)
            # print('r1_x_r2_norm_sq', r1_x_r2_norm_sq.shape)
            # print('r1_norm', r1_norm.shape)
            # print('r2_r1_norm_sq', r2_r1_norm_sq.shape)
            # print('rc_sq', rc_sq.shape)
            # print('array1', array1.shape)
            # pertubation = csdl.expand(r2_r1_norm_sq * rc_sq_expand,
            #                           array1.shape, 'i->ik')
            # print('pertubation', pertubation.shape)
            mesh_resolution = 1

            # array2 = ((csdl.einsum(
            #     (r1 * r2_norm - r2 * r1_norm) /
            #     (r1_norm * r2_norm +
            #      (r1_norm**2 + r2_norm**2) * self.parameters['eps']),
            #     r0,
            #     subscripts='ij,ij->i',
            # )))

            array2 = ((csdl.einsum(
                (r1 * r2_norm - r2 * r1_norm) /
                (r1_norm * r2_norm + mesh_resolution * self.parameters['eps']),
                r0,
                subscripts='kij,kij->ki',
                partial_format='sparse',
            )))
            # pertubation = 0.01219
            # v_induced_line = array1 * csdl.expand(
            #     array2, array1.shape, 'i->ij') / (
            #         r1_x_r2_norm_sq + pertubation * self.parameters['eps']
            #     )  # TODO: fix this later

            v_induced_line = array1 * csdl.expand(
                array2, array1.shape, 'ki->kij') / (
                    r1_x_r2_norm_sq + 1 * self.parameters['eps']
                )  # TODO: fix this later

            # r1_dot_r2 = csdl.dot(r1, r2, axis=1)
            # r1_dot_r2_expand = csdl.expand(r1_dot_r2,
            #                                (r1_dot_r2.shape) + (3, ), 'i->ik')
            # v_induced_line = 1 / (np.pi * 4) * r1_x_r2 * (
            #     (r1_norm + r2_norm) * (r1_norm * r2_norm - r1_dot_r2_expand) /
            #     (r1_norm * r2_norm * (r1_x_r2_norm_sq + pertubation * 1e-3))
            # )
        else:
            array2 = ((csdl.einsum(
                (r1 * r2_norm - r2 * r1_norm) / (r1_norm * r2_norm),
                r0,
                subscripts='kij,kij->ki',
                partial_format='sparse',
            )))
            v_induced_line = array1 * csdl.expand(
                array2, array1.shape, 'ki->kij') / (r1_x_r2_norm_sq)

        return v_induced_line


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

    n_wake_pts_chord = 6
    nx = 3
    ny = 4

    eval_pt_names = ['col']
    vortex_coords_names = ['vor']
    # eval_pt_shapes = [(nx, ny, 3)]
    # vortex_coords_shapes = [(nx, ny, 3)]

    eval_pt_shapes = [(2, 3, 3)]
    vortex_coords_shapes = [(nx, ny, 3)]

    output_names = ['aic']

    model_1 = Model()

    # circulations_val = np.zeros(
    #     (nx - 1, ny - 1))  ####################the size of this is important
    # circulations_val[:2, :] = np.random.random((2, ny - 1))
    circulations_val = np.ones((nx - 1, ny - 1)) * 0.5

    vor_val = generate_simple_mesh(nx, ny)
    col_val = 0.25 * (vor_val[:-1, :-1, :] + vor_val[:-1, 1:, :] +
                      vor_val[1:, :-1, :] + vor_val[1:, 1:, :])
    # col_val = generate_simple_mesh(nx, ny)

    vor = model_1.create_input('vor', val=vor_val)
    col = model_1.create_input('col', val=col_val)
    circulations = model_1.create_input('circulations',
                                        val=circulations_val.reshape(
                                            1, nx - 1, ny - 1))

    model_1.add(BiotSavartComp(eval_pt_names=eval_pt_names,
                               vortex_coords_names=vortex_coords_names,
                               eval_pt_shapes=eval_pt_shapes,
                               vortex_coords_shapes=vortex_coords_shapes,
                               output_names=output_names,
                               vc=True,
                               n_wake_pts_chord=n_wake_pts_chord,
                               circulation_names=['circulations']),
                name='BiotSvart_group')
    sim = Simulator(model_1)

    print(sim['vor'])
    print(sim[output_names[0]])
    # sim.visualize_implementation()
    sim.run()

    a_l = 1.25643
    kinematic_viscocity = 1.48 * 1e-5
    a_1 = 0.1
    time_current = 2
    sigma = 1 + a_1 * csdl.reshape(
        circulations, new_shape=(circulations.shape[1:])) / kinematic_viscocity
    r_c = (4 * a_l * kinematic_viscocity * sigma.val * time_current)**0.5
