# from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size


class WakeCoords(Model):
    """
    compute wake vortex coords given the vortex coords

    parameters
    ----------
    frame_vel[num_nodes,3] : csdl array
        inertia frame vel
    bd_vtx_coords[num_nodes,num_pts_chord, num_pts_span, 3] : csdl array
    bound vortices points

    Returns
    -------
    1. wake_coords[num_nodes, 2, num_pts_span-1, 3] : csdl array
        wake vortex coordianates    
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('n_wake_pts_chord')
        self.parameters.declare('delta_t')
        self.parameters.declare('TE_idx', default='last')

    def define(self):
        # add_input
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = surface_shapes[0][0]
        n_wake_pts_chord = self.parameters[
            'n_wake_pts_chord']  # number of wake nodes in streamwise direction
        delta_t = self.parameters['delta_t']
        TE_idx = self.parameters['TE_idx']

        wake_coords_names = [x + '_wake_coords' for x in surface_names]

        frame_vel = self.declare_variable('frame_vel', shape=(num_nodes, 3))

        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            surface_shape = surface_shapes[i]
            bd_vtx_coords_shape = surface_shape
            bd_vtx_coords_name = surface_name + '_bd_vtx_coords'

            num_pts_chord = surface_shape[1]
            num_pts_span = surface_shape[2]

            bd_vtx_coords = self.declare_variable(bd_vtx_coords_name,
                                                  shape=bd_vtx_coords_shape)
            if TE_idx == 'first':
                TE = bd_vtx_coords[:, 0, :, :]
            elif TE_idx == 'last':
                TE = bd_vtx_coords[:, num_pts_chord - 1, :, :]

            TE_reshaped = csdl.reshape(TE, (num_nodes, num_pts_span, 3))
            TE_reshaped_expand = csdl.expand(
                TE_reshaped, (num_nodes, n_wake_pts_chord, num_pts_span, 3),
                'ijk->iljk')
            # print('TE_reshaped_expand shape', TE_reshaped_expand.shape)

            factor_var = np.einsum('i,jkl->jikl',
                                   np.arange(n_wake_pts_chord) * delta_t,
                                   np.ones((num_nodes, num_pts_span, 3)))
            factor = self.create_input(surface_name + '_factor',
                                       val=factor_var)
            #! TODO:! fix this for rotating surfaces
            # - should be fine actually just to align the wake w/ free stream
            delta_x = csdl.expand(-frame_vel,
                                  (num_nodes, n_wake_pts_chord, num_pts_span,
                                   3), 'il->ijkl') * factor
            wake_coords = TE_reshaped_expand + delta_x

            self.register_output(wake_coords_names[i], wake_coords)


if __name__ == "__main__":
    import csdl_lite
    # simulator_name = 'csdl_om'
    simulator_name = 'csdl_lite'

    n_wake_pts_chord = 2
    num_pts_chord = 3
    num_pts_span = 4
    num_nodes = 2

    def generate_simple_mesh(num_nodes, num_pts_chord, num_pts_span, offset=0):
        mesh = np.zeros((num_nodes, num_pts_chord, num_pts_span, 3))
        for i in range(num_nodes):
            mesh[i, :, :, 0] = np.outer(np.arange(num_pts_chord),
                                        np.ones(num_pts_span))
            mesh[i, :, :, 1] = np.outer(np.arange(num_pts_span),
                                        np.ones(num_pts_chord)).T + offset
            mesh[i, :, :, 2] = 0.
        return mesh

    surface_names = ['wing_1', 'wing_2']
    surface_shapes = [(num_nodes, num_pts_chord, num_pts_span, 3),
                      (num_nodes, num_pts_chord + 1, num_pts_span + 1, 3)]
    model_1 = Model()

    wing_1_mesh = generate_simple_mesh(num_nodes, num_pts_chord, num_pts_span)
    wing_2_mesh = generate_simple_mesh(num_nodes,
                                       num_pts_chord + 1,
                                       num_pts_span + 1,
                                       offset=10)

    f = model_1.create_input('frame_vel',
                             val=np.array([[-1, 0, -1], [-1, 0, -1]]))

    wing_1_inputs = model_1.create_input('wing_1_bd_vtx_coords',
                                         val=wing_1_mesh)
    wing_2_inputs = model_1.create_input('wing_2_bd_vtx_coords',
                                         val=wing_2_mesh)

    model_1.add(WakeCoords(
        surface_names=surface_names,
        surface_shapes=surface_shapes,
        n_wake_pts_chord=n_wake_pts_chord,
        delta_t=20,
    ),
                name='WakeCoords')

    if simulator_name == 'csdl_om':

        sim = Simulator(model_1)

        sim.run()
        # sim.prob.check_partials(compact_print=True)
        partials = sim.prob.check_partials(compact_print=True, out_stream=None)
        sim.assert_check_partials(partials, 1e-5, 1e-7)
        sim.visualize_implementation()
        sim.prob.check_config(checks=['unconnected_inputs'], out_file=None)

    elif simulator_name == 'csdl_lite':
        sim = csdl_lite.Simulator(model_1)

        sim.run()
        sim.check_partials(compact_print=True)

    # import pyvista as pv
    # ############################################
    # # Plot the lifting surfaces
    # ############################################
    # pv.global_theme.axes.show = True
    # pv.global_theme.font.label_size = 1
    # x = wing_1_mesh[0, :, :, 0]
    # y = wing_1_mesh[0, :, :, 1]
    # z = wing_1_mesh[0, :, :, 2]
    # x_1 = wing_2_mesh[0, :, :, 0]
    # y_1 = wing_2_mesh[0, :, :, 1]
    # z_1 = wing_2_mesh[0, :, :, 2]

    # xw = sim['wing_1_wake_coords'][0, :, :, 0]
    # yw = sim['wing_1_wake_coords'][0, :, :, 1]
    # zw = sim['wing_1_wake_coords'][0, :, :, 2]

    # xw_1 = sim['wing_2_wake_coords'][0, :, :, 0]
    # yw_1 = sim['wing_2_wake_coords'][0, :, :, 1]
    # zw_1 = sim['wing_2_wake_coords'][0, :, :, 2]

    # grid = pv.StructuredGrid(x, y, z)
    # grid_1 = pv.StructuredGrid(x_1, y_1, z_1)
    # gridw = pv.StructuredGrid(xw, yw, zw)
    # gridw_1 = pv.StructuredGrid(xw_1, yw_1, zw_1)
    # p = pv.Plotter()
    # p.add_mesh(grid, color="blue", show_edges=True, opacity=.5)
    # p.add_mesh(gridw, color="blue", show_edges=True, opacity=.5)
    # p.add_mesh(grid_1, color="red", show_edges=True, opacity=.5)
    # p.add_mesh(gridw_1, color="red", show_edges=True, opacity=.5)
    # p.camera.view_angle = 60.0
    # p.add_axes_at_origin(labels_off=True, line_width=5)

    # p.show()