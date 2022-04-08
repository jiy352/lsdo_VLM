from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size


class MeshPreprocessing(Model):
    """
    Compute various geometric properties for VLM analysis.
    These are used primarily to help compute postprocessing quantities,
    such as wave CD, viscous CD, etc.
    Some of the quantities, like `normals`, are used to compute the RHS
    of the AIC linear system.

    parameters
    ----------
    def_mesh[nx, ny, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface.

    Returns
    -------
    # b_pts[nx, ny, 3] : csdl array
    #     Bound points for the horseshoe vortices, found along the 1/4 chord.
    bd_vtx_coords[nx, ny, 3] : csdl array
    bound vortices points
    coll_pts[nx-1, ny-1, 3] : csdl array
        collocation points for the horseshoe vortices, found along the 3/4 chord.
    l_span[(nx-1)* (ny-1)] : csdl array
        The spanwise widths of each individual panel.
    l_chord[(nx-1)* (ny-1)] : csdl array
        The chordwise length of of each individual panel.
    s_panel [(nx-1)* (ny-1)]: csdl array
        The panel areas.
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('dynamic_parameter', default=False)

    def define(self):
        # add_input
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        dynamic_parameter = self.parameters['dynamic_parameter']
        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            bd_vtx_coords_name = surface_name + '_bd_vtx_coords'
            coll_pts_coords_name = surface_name + '_coll_pts_coords'
            chord_name = surface_name + '_chord_length'
            span_name = surface_name + '_span_length'
            s_panel_name = surface_name + '_s_panel'

            nx = surface_shapes[i][1]
            ny = surface_shapes[i][2]

            if dynamic_parameter == True:

                def_mesh = self.declare_variable(surface_name,
                                                 shape=surface_shapes[i])
                bd_vtx_coords = self.create_output(bd_vtx_coords_name,
                                                   shape=(def_mesh.shape))
                bd_vtx_coords[:, 0:nx -
                              1, :, :] = def_mesh[:, 0:nx -
                                                  1, :, :] * .75 + def_mesh[:,
                                                                            1:
                                                                            nx, :, :] * 0.25
                bd_vtx_coords[:, nx -
                              1, :, :] = def_mesh[:, nx - 1, :, :] + 0.25 * (
                                  def_mesh[:, nx - 1, :, :] -
                                  def_mesh[:, nx - 2, :, :])
                coll_pts_coords = 0.25 * (bd_vtx_coords[:,0:nx-1, 0:ny-1, :] +\
                                                bd_vtx_coords[:,0:nx-1, 1:ny, :] +\
                                                bd_vtx_coords[:,1:, 0:ny-1, :]+\
                                                bd_vtx_coords[:,1:, 1:, :])
                self.register_output(coll_pts_coords_name, coll_pts_coords)

                chords_vec = def_mesh[:, 0:nx - 1, :, :] - def_mesh[:,
                                                                    1:, :, :]
                chords = csdl.pnorm(chords_vec, axis=(3))
                # self.register_output(chord_name, chords)

                span_vec = def_mesh[:, :, 0:ny - 1, :] - def_mesh[:, :, 1:, :]
                spans = csdl.pnorm(span_vec, axis=(2))
                # self.register_output(span_name, spans)
                # TODO: need to fix this before computing the forces

                i = def_mesh[:, :-1, 1:, :] - def_mesh[:, 1:, :-1, :]
                j = def_mesh[:, :-1, :-1, :] - def_mesh[:, 1:, 1:, :]
                normals = csdl.cross(i, j, axis=3)
                s_panel = (csdl.sum(normals**2, axes=(3, )))**0.5 * 0.5
                # area = |ixj|/2
                # self.register_output(s_panel_name, s_panel)

            else:
                def_mesh = self.declare_variable(surface_name,
                                                 shape=surface_shapes[i])
                bd_vtx_coords = self.create_output(bd_vtx_coords_name,
                                                   shape=(nx, ny, 3))
                bd_vtx_coords[0:nx - 1, :, :] = csdl.reshape(
                    def_mesh[:, 0:nx - 1, :, :] * .75 +
                    def_mesh[:, 1:nx, :, :] * 0.25, (nx - 1, ny, 3))
                bd_vtx_coords[nx - 1, :, :] = csdl.reshape(
                    def_mesh[:, nx - 1, :, :] + 0.25 *
                    (def_mesh[:, nx - 1, :, :] - def_mesh[:, nx - 2, :, :]),
                    (1, ny, 3))
                coll_pts_coords = 0.25 * (bd_vtx_coords[0:nx-1, 0:ny-1, :] +\
                                        bd_vtx_coords[0:nx-1, 1:ny, :] +\
                                                bd_vtx_coords[1:, 0:ny-1, :]+\
                                                bd_vtx_coords[1:, 1:, :])
                self.register_output(coll_pts_coords_name, coll_pts_coords)


if __name__ == "__main__":
    nt = 2
    nx = 3
    ny = 4

    def generate_simple_mesh(nt, nx, ny):
        mesh = np.zeros((nt, nx, ny, 3))
        for i in range(nt):
            mesh[i, :, :, 0] = np.outer(np.arange(nx), np.ones(ny))
            mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
            mesh[i, :, :, 2] = 0.
        return mesh

    surface_names = ['wing_1', 'wing_2']
    surface_shapes = [(nt, nx, ny, 3), (nt, nx + 1, ny + 1, 3)]
    model_1 = Model()

    wing_1_mesh = generate_simple_mesh(nt, nx, ny)
    wing_2_mesh = generate_simple_mesh(nt, nx + 1, ny + 1)

    wing_1_inputs = model_1.create_input('wing_1', val=wing_1_mesh)
    wing_2_inputs = model_1.create_input('wing_2', val=wing_2_mesh)
    # model_1.register_output('wing_1', wing_1_inputs + 0)
    # model_1.register_output('wing_2', wing_2_inputs + 0)
    model_1.add(MeshPreprocessing(surface_names=surface_names,
                                  surface_shapes=surface_shapes),
                name='preprocessing_group')

    sim = Simulator(model_1)
    # sim.prob.set_val('wing_1', val=wing_1_mesh)
    # sim.prob.set_val('wing_2', val=wing_2_mesh)
    # sim.visualize_implementation()
    sim.run()
    import matplotlib.pyplot as plt

    plt.plot(sim[surface_names[0] + '_bd_vtx_coords'][0, :, :, 0].flatten(),
             sim[surface_names[0] + '_bd_vtx_coords'][0, :, :,
                                                      1].flatten(), 'ro')
    plt.plot(sim[surface_names[0] + '_coll_pts_coords'][0, :, :, 0].flatten(),
             sim[surface_names[0] + '_coll_pts_coords'][0, :, :,
                                                        1].flatten(), 'bo')
    plt.legend(['bs', 'coll'])

    # x, y = np.meshgrid(np.linspace(0, nx - 1, nx), np.linspace(0, ny - 1, ny))
    # plt.plot(x, y, 'b-')
    # plt.plot(x.T, y.T, 'b-')
    plt.axis('equal')
    plt.show()
    # sim.visualize_model()

    # x, y = np.meshgrid(np.linspace(0, nx - 1, nx), np.linspace(0, ny - 1, ny))
    # plt.plot(x, y, 'b-')
    # plt.plot(x.T, y.T, 'b-')

    # nx = 20
    # ny = 4
    # x, y = np.meshgrid(np.linspace(2, nx - 1, nx), np.linspace(0, ny - 1, ny))
    # plt.plot(x, y, 'r-')
    # plt.plot(x.T, y.T, 'r-')
    # plt.legend(['bound vortices', 'wake vortices'])
    # plt.axis('equal')
    # plt.show()