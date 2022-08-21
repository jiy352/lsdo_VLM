# from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size
# from UVLM_package.VLM_system.solve_circulations.biot_savart_comp_org import BiotSavartComp
from VLM_package.VLM_system.solve_circulations.biot_savart_vc_comp import BiotSavartComp


class AssembleAic(Model):
    """
    Compute various geometric properties for VLM analysis.
    These are used primarily to help compute postprocessing quantities,
    such as wave CD, viscous CD, etc.
    Some of the quantities, like `normals`, are used to compute the RHS
    of the AIC linear system.

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
        # aic_col_w
        self.parameters.declare('bd_coll_pts_names', types=list)
        self.parameters.declare('wake_vortex_pts_names', types=list)

        self.parameters.declare('bd_coll_pts_shapes', types=list)
        self.parameters.declare('wake_vortex_pts_shapes', types=list)
        self.parameters.declare('full_aic_name', types=str)

        self.parameters.declare('delta_t')

    def define(self):
        # add_input
        bd_coll_pts_names = self.parameters['bd_coll_pts_names']
        wake_vortex_pts_names = self.parameters['wake_vortex_pts_names']
        delta_t = self.parameters['delta_t']

        bd_coll_pts_shapes = self.parameters['bd_coll_pts_shapes']
        wake_vortex_pts_shapes = self.parameters['wake_vortex_pts_shapes']
        full_aic_name = self.parameters['full_aic_name']
        num_nodes = bd_coll_pts_shapes[0][0]
        row_ind = 0
        col_ind = 0

        eval_pt_names = []
        vortex_coords_names = []
        eval_pt_shapes = []
        vortex_coords_shapes = []
        output_names = []
        aic_shape_row = aic_shape_col = 0

        for i in range(len(bd_coll_pts_shapes)):

            bd_coll_pts = self.declare_variable(bd_coll_pts_names[i],
                                                shape=bd_coll_pts_shapes[i])
            wake_vortex_pts = self.declare_variable(
                wake_vortex_pts_names[i], shape=wake_vortex_pts_shapes[i])
            aic_shape_row += (bd_coll_pts_shapes[i][1] *
                              bd_coll_pts_shapes[i][2])
            aic_shape_col += ((wake_vortex_pts_shapes[i][1] - 1) *
                              (wake_vortex_pts_shapes[i][2] - 1))

            for j in range(len(wake_vortex_pts_shapes)):
                eval_pt_names.append(bd_coll_pts_names[i])
                vortex_coords_names.append(wake_vortex_pts_names[j])
                eval_pt_shapes.append(bd_coll_pts_shapes[i])
                vortex_coords_shapes.append(wake_vortex_pts_shapes[j])
                out_name = full_aic_name + str(i) + str(j)
                output_names.append(out_name)

        # print('assemble_aic line 84 eval_pt_names', eval_pt_names)
        # print('assemble_aic line 85 vortex_coords_names', vortex_coords_names)
        # print('assemble_aic line 86 eval_pt_shapes', eval_pt_shapes)
        # print('assemble_aic l 87 vortex_coords_shapes', vortex_coords_shapes)
        # print('assemble_aic l 87 output_names', output_names)

        m = BiotSavartComp(
            eval_pt_names=eval_pt_names,
            vortex_coords_names=vortex_coords_names,
            eval_pt_shapes=eval_pt_shapes,
            vortex_coords_shapes=vortex_coords_shapes,
            output_names=output_names,
        )
        self.add(m, name='aic_bd_w_seperate')

        aic_shape = (num_nodes, aic_shape_row, aic_shape_col, 3)

        # m1 = Model()
        aic_col_w = self.create_output(full_aic_name, shape=aic_shape)
        row = 0
        col = 0
        for i in range(len(bd_coll_pts_shapes)):
            for j in range(len(wake_vortex_pts_shapes)):
                aic_i_shape = (
                    num_nodes,
                    bd_coll_pts_shapes[i][1] * bd_coll_pts_shapes[i][2] *
                    (wake_vortex_pts_shapes[j][1] - 1) *
                    (wake_vortex_pts_shapes[j][2] - 1),
                    3,
                )
                aic_i = self.declare_variable(
                    output_names[i * (len(wake_vortex_pts_shapes)) + j],
                    shape=aic_i_shape)

                delta_row = bd_coll_pts_shapes[i][1] * bd_coll_pts_shapes[i][2]
                delta_col = (wake_vortex_pts_shapes[j][1] -
                             1) * (wake_vortex_pts_shapes[j][2] - 1)

                aic_col_w[:, row:row + delta_row,
                          col:col + delta_col, :] = csdl.reshape(
                              aic_i,
                              new_shape=(num_nodes, delta_row, delta_col, 3))
                col = col + delta_col
            col = 0
            row = row + delta_row


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

    bd_coll_pts_names = ['bd_coll_pts_1', 'bd_coll_pts_2']
    wake_vortex_pts_names = ['wake_vortex_pts_1', 'wake_vortex_pts_2']
    bd_coll_pts_shapes = [(2, 3, 3), (3, 2, 3)]
    wake_vortex_pts_shapes = [(3, 3, 3), (3, 2, 3)]
    # bd_coll_pts_shapes = [(2, 3, 3), (2, 3, 3)]
    # wake_vortex_pts_shapes = [(3, 3, 3), (3, 3, 3)]

    model_1 = Model()
    bd_val = np.random.random((2, 3, 3))
    bd_val_1 = np.random.random((3, 2, 3))
    wake_vortex_val = np.random.random((3, 3, 3))
    wake_vortex_val_1 = np.random.random((3, 2, 3))

    # bd_val = np.random.random((2, 3, 3))
    # bd_val_1 = np.random.random((2, 3, 3))
    # wake_vortex_val = np.random.random((3, 3, 3))
    # wake_vortex_val_1 = np.random.random((3, 3, 3))

    bd = model_1.create_input('bd_coll_pts_1', val=bd_val)
    bd_1 = model_1.create_input('bd_coll_pts_2', val=bd_val_1)

    wake_vortex = model_1.create_input('wake_vortex_pts_1',
                                       val=wake_vortex_val)
    wake_vortex_1 = model_1.create_input('wake_vortex_pts_2',
                                         val=wake_vortex_val_1)
    model_1.add(AssembleAic(
        bd_coll_pts_names=bd_coll_pts_names,
        wake_vortex_pts_names=wake_vortex_pts_names,
        bd_coll_pts_shapes=bd_coll_pts_shapes,
        wake_vortex_pts_shapes=wake_vortex_pts_shapes,
    ),
                name='assemble_aic_comp')
    sim = Simulator(model_1)
    sim.run()
    # sim.visualize_implementation()
    # print('aic is', sim['aic'])
    # print('v_ind is', sim['v_ind'].shape, sim['v_ind'])
