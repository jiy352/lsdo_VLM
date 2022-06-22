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


class ComputeNormal(Model):
    """
    Compute normals.

    parameters
    ----------
    vortex_coords[nx, ny, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface.

    Returns
    -------
    normals[nx-1, ny-1, 3] : numpy array
        The normals of each vortex panel
    """
    def initialize(self):
        self.parameters.declare('vortex_coords_names', types=list)
        self.parameters.declare('normals_names', types=list)

        self.parameters.declare('vortex_coords_shapes', types=list)

    def define(self):
        vortex_coords_names = self.parameters['vortex_coords_names']
        normals_names = self.parameters['normals_names']

        vortex_coords_shapes = self.parameters['vortex_coords_shapes']
        # print('compute_normal line 44 vortex_coords_shapes',
        #       vortex_coords_shapes)

        for i in range(len(vortex_coords_names)):

            # input_names
            vortex_coords_name = vortex_coords_names[i]

            # output_name
            normals_name = normals_names[i]
            # input_shapes
            vortex_coords_shape = vortex_coords_shapes[i]

            # declare_inputs
            vortex_coords = self.declare_variable(vortex_coords_name,
                                                  shape=vortex_coords_shape)
            i = vortex_coords[:, :-1, 1:, :] - vortex_coords[:, 1:, :-1, :]
            j = vortex_coords[:, :-1, :-1, :] - vortex_coords[:, 1:, 1:, :]
            normals = csdl.cross(i, j, axis=3)
            norms = (csdl.sum(normals**2, axes=(3, )))**0.5
            norms_expanded = csdl.expand(norms, norms.shape + (3, ),
                                         'ijk->ijkl')
            normals_normalized = normals / norms_expanded

            self.register_output(normals_name, normals_normalized)


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

    num_nodes = 3
    vortex_coords_names = ['v1', 'v2']
    normals_names = ['n1', 'n2']
    vortex_coords_shapes = [(num_nodes, 2, 3, 3), (num_nodes, 3, 4, 3)]

    model_1 = Model()
    v1_val = generate_simple_mesh(2, 3, num_nodes)
    v2_val = generate_simple_mesh(3, 4, num_nodes)

    v1 = model_1.create_input('v1', val=v1_val)
    v2 = model_1.create_input('v2', val=v2_val)
    model_1.add(ComputeNormal(
        vortex_coords_names=vortex_coords_names,
        normals_names=normals_names,
        vortex_coords_shapes=vortex_coords_shapes,
    ),
                name='ComputeNormal')
    sim = Simulator(model_1)
    # sim.visualize_implementation()
    sim.run()
