# from csdl_om import Simulator
from csdl import Model
import csdl
from matplotlib.pyplot import axis, clabel
import numpy as np
from numpy.core.fromnumeric import size

from scipy.sparse import csc_matrix
from VLM_package.VLM_system.solve_circulations.combine_bd_wake_comp import BdnWakeCombine
from VLM_package.VLM_system.solve_circulations.biot_savart_vc_comp import BiotSavartComp
from VLM_package.VLM_system.solve_circulations.induced_velocity_comp import InducedVelocity


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

        self.parameters.declare('eval_pts_option')
        self.parameters.declare('eval_pts_location')
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        # stands for quarter-chord
        self.parameters.declare('n_wake_pts_chord')
        self.parameters.declare('delta_t')
        self.parameters.declare('mesh_unit', default='m')

    def define(self):
        # eval_pts_names = self.parameters['eval_pts_names']
        eval_pts_shapes = self.parameters['eval_pts_shapes']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        eval_pts_location = self.parameters['eval_pts_location']
        mesh_unit = self.parameters['mesh_unit']
        eval_pts_option = self.parameters['eval_pts_option']

        num_nodes = surface_shapes[0][0]

        n_wake_pts_chord = self.parameters['n_wake_pts_chord']
        delta_t = self.parameters['delta_t']

        bdnwake_coords_names = [x + '_bdnwake_coords' for x in surface_names]
        if eval_pts_option=='auto':
            eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]

        else:
            eval_pts_names=self.parameters['eval_pts_names']

        eval_induced_velocities_col_names = [
            x + '_eval_pts_induced_vel_col' for x in eval_pts_names
        ]

        wake_coords_reshaped_names = [
            x + '_wake_coords_reshaped' for x in surface_names
        ]

        wake_vortex_pts_shapes = [
            tuple((num_nodes, n_wake_pts_chord, item[2], 3))
            for item in surface_shapes
        ]

        bdnwake_shapes = [
            (num_nodes, x[1] + y[1] - 1, x[2], 3)
            for x, y in zip(surface_shapes, wake_vortex_pts_shapes)
        ]
        # output_names = [x + '_aic_force' for x in surface_names]
        circulation_names = [x + '_bdnwake_gamma' for x in surface_names]
        # print('eval pts vel bdnwake_shapes ', bdnwake_shapes)
        # print('eval pts vel eval_pts_shapes ', eval_pts_shapes)

        aic_shapes = [(num_nodes, x[1] * x[2] * (y[1] - 1) * (y[2] - 1), 3)
                      for x, y in zip(eval_pts_shapes, bdnwake_shapes)]

        circulations_shapes = [
            (num_nodes, (x[1] - 1) * (x[2] - 1) + (y[1] - 1) * (y[2] - 1))
            for x, y in zip(surface_shapes, wake_vortex_pts_shapes)
        ]
        eval_induced_velocities_names = [
            x + '_eval_pts_induced_vel' for x in eval_pts_names
        ]


        if eval_pts_option=='auto':
            eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
        else:
            eval_pts_names=self.parameters['eval_pts_names']

        # v_total_eval_names = [x + '_eval_total_vel' for x in surface_names]
        v_total_eval_names = [x + '_eval_total_vel' for x in eval_pts_names]

        eval_vel_shapes = [(num_nodes, x[1] * x[2], 3)
                           for x in eval_pts_shapes]

        #!TODO!: rewrite this comp for mls
        # !fixed!: defining the eval_pts


        if self.parameters['eval_pts_option'] == 'auto':
            for i in range(len(surface_shapes)):
                if mesh_unit == 'm':
                    mesh = self.declare_variable(surface_names[i],
                                                shape=surface_shapes[i])
                elif mesh_unit == 'ft':
                    mesh_ft = self.declare_variable(surface_names[i],
                                                    shape=surface_shapes[i])
                    mesh = mesh_ft * 0.3048

                eval_pts_coords = (
                    (1 - eval_pts_location) * 0.5 * mesh[:, 0:-1, 0:-1, :] +
                    (1 - eval_pts_location) * 0.5 * mesh[:, 0:-1, 1:, :] +
                    eval_pts_location * 0.5 * mesh[:, 1:, 0:-1, :] +
                    eval_pts_location * 0.5 * mesh[:, 1:, 1:, :])

                self.register_output(eval_pts_names[i], eval_pts_coords)

        elif self.parameters['eval_pts_option'] == 'user_defined':
            for i in range(len(eval_pts_shapes)):
                eval_pts_coords = self.declare_variable(
                    eval_pts_names[i], shape=(eval_pts_shapes[i]))

        self.add(BdnWakeCombine(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            n_wake_pts_chord=n_wake_pts_chord,
        ),
                 name='BdnWakeCombine')

        for i in range(len(surface_shapes)):
            nx = surface_shapes[i][1]
            ny = surface_shapes[i][2]
            bdnwake_coords = self.declare_variable(
                bdnwake_coords_names[i],
                shape=(num_nodes, n_wake_pts_chord + nx - 1, ny, 3))
        #!TODO:fix this for mls
        # !fixed!: this part is a temp fix-since we don't have +=in csdl, I just made a large velocity matrix contining
        # the induced velocity induced by each bdnwake_coords_names for mls, and sum this matrix by axis to get the
        # total induced vel
        for i in range(len(eval_pts_names)):
            eval_vel_shape = eval_vel_shapes[i]

            aic_shapes = [
                (num_nodes, x[1] * x[2] * (y[1] - 1) * (y[2] - 1), 3)
                for x, y in zip(([eval_pts_shapes[i]] *
                                 len(bdnwake_coords_names)), bdnwake_shapes)
            ]
            eval_pts_name_repeat = [eval_pts_names[i]
                                    ] * len(bdnwake_coords_names)

            output_names = [
                eval_pts_names[i] + x + '_out' for x in bdnwake_coords_names
            ]
            induced_vel_bdnwake_names = [
                eval_pts_names[i] + x + '_induced_vel'
                for x in bdnwake_coords_names
            ]
            self.add(BiotSavartComp(
                eval_pt_names=eval_pts_name_repeat,
                vortex_coords_names=bdnwake_coords_names,
                eval_pt_shapes=[eval_pts_shapes[i]] *
                len(bdnwake_coords_names),
                vortex_coords_shapes=bdnwake_shapes,
                output_names=output_names,
                circulation_names=circulation_names,
                vc=True,
            ),
                     name='eval_pts_aics' + str(i))

            for j in range(len(bdnwake_coords_names)):
                aic = self.declare_variable(output_names[j],
                                            shape=(aic_shapes[j]))
            # print('eval pts vel mls aic_shapes', aic_shapes)

            self.add(InducedVelocity(
                aic_names=output_names,
                circulation_names=circulation_names,
                aic_shapes=aic_shapes,
                circulations_shapes=circulations_shapes,
                v_induced_names=induced_vel_bdnwake_names),
                     name='eval_pts_ind_vel' + str(i))
            # !!!!!!!!!!!TODO: need to check what is this April 18 2022
            # 08/03 this is used to add up the induced velocites
            surface_total_induced_col = self.create_output(
                eval_induced_velocities_col_names[i],
                shape=(num_nodes, len(bdnwake_coords_names),
                       eval_vel_shapes[i][1], 3))
            for j in range(len(bdnwake_coords_names)):
                induced_vel_bdnwake = self.declare_variable(
                    induced_vel_bdnwake_names[j],
                    shape=(num_nodes, eval_vel_shape[1], 3))
                surface_total_induced_col[:, j, :, :] = csdl.reshape(
                    induced_vel_bdnwake,
                    (num_nodes, 1, eval_vel_shapes[i][1], 3))
            eval_induced_velocity = self.register_output(
                eval_induced_velocities_names[i],
                csdl.sum(surface_total_induced_col, axes=(1, )))

        # kinematic_vel_names = [
        #     x + '_kinematic_vel' for x in self.parameters['surface_names']
        # ]
        # TODO: check this part for the whole model
        model_wake_total_vel = Model()
        for i in range(len(eval_induced_velocities_names)):
            v_induced_wake_name = eval_induced_velocities_names[i]
            eval_vel_shape = eval_vel_shapes[i]

            # kinematic_vel_name = kinematic_vel_names[i]

            v_induced_wake = model_wake_total_vel.declare_variable(
                v_induced_wake_name, shape=eval_vel_shape)

            # !!TODO!! this needs to be fixed for more general cases to compute the undisturbed vel
            # Note - April 7 2022: the wake velocity seems to just
            # need to be the sum of free stream and the induced velocity - this part seems to be fine for now


            frame_vel = model_wake_total_vel.declare_variable('frame_vel',
                                                              shape=(num_nodes,
                                                                     3))
            frame_vel_expand = csdl.expand(frame_vel,
                                           eval_vel_shape,
                                           indices='li->lji')

            v_total_wake = csdl.reshape((v_induced_wake - frame_vel_expand),
                                        new_shape=eval_vel_shape)

            model_wake_total_vel.register_output(v_total_eval_names[i],
                                                 v_total_wake)

        self.add(model_wake_total_vel, name='eval_pts_total_vel')
