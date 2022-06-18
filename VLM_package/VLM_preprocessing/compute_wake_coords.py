from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size


class WakeCoords(Model):
    """
    compute wake vortex coords given the vortex coords
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('num_nodes', types=int)
        self.parameters.declare('nt')
        self.parameters.declare('delta_t')

    def define(self):
        # add_input
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = self.parameters['num_nodes']
        nt = self.parameters['nt']
        delta_t = self.parameters['delta_t']

        wake_coords_names = [x + '_wake_coords' for x in surface_names]

        frame_vel = self.declare_variable('frame_vel', shape=(num_nodes, 3))

        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            surface_shape = surface_shapes[i]
            bd_vtx_coords_shape = surface_shape
            bd_vtx_coords_name = surface_name + '_bd_vtx_coords'

            nx = surface_shape[1]
            ny = surface_shape[2]

            bd_vtx_coords = self.declare_variable(bd_vtx_coords_name,
                                                  shape=bd_vtx_coords_shape)
            TE_pos = 'last'
            if TE_pos == 'first':
                TE = bd_vtx_coords[:, 0, :, :]
            else:
                TE = bd_vtx_coords[:, nx - 1, :, :]
            # print('TE shape', TE.shape)
            TE_reshaped = csdl.reshape(TE, (num_nodes, ny, 3))
            TE_reshaped_expand = csdl.expand(TE_reshaped,
                                             (num_nodes, nt, ny, 3),
                                             'ijk->iljk')
            # print('TE_reshaped_expand shape', TE_reshaped_expand.shape)

            factor_var = np.einsum('i,jkl->jikl',
                                   np.arange(nt) * delta_t,
                                   np.ones((num_nodes, ny, 3)))
            factor = self.create_input(surface_name + '_factor',
                                       val=factor_var)
            #! TODO:! fix this for rotating surfaces
            # - should be fine actually just to align the wake w/ free stream
            delta_x = csdl.expand(-frame_vel,
                                  (num_nodes, nt, ny, 3), 'il->ijkl') * factor
            wake_coords = TE_reshaped_expand + delta_x
            # print('wake_coords shape', wake_coords.shape)

            self.register_output(wake_coords_names[i], wake_coords)


if __name__ == "__main__":
    nt = 5
    nx = 3
    ny = 4

    def generate_simple_mesh(nx, ny):
        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))
        mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
        mesh[:, :, 2] = 0.
        return mesh

    surface_names = ['wing_1', 'wing_2']
    surface_shapes = [(1, nx, ny, 3), (1, nx + 1, ny + 1, 3)]
    model_1 = Model()

    wing_1_mesh = generate_simple_mesh(nx, ny)
    wing_2_mesh = generate_simple_mesh(nx + 1, ny + 1)

    f = model_1.create_input('frame_vel', val=np.array([-1, 0, -1]))

    wing_1_inputs = model_1.create_input('wing_1_bd_vtx_coords',
                                         val=wing_1_mesh)
    wing_2_inputs = model_1.create_input('wing_2_bd_vtx_coords',
                                         val=wing_2_mesh)
    # model_1.register_output('wing_1', wing_1_inputs + 0)
    # model_1.register_output('wing_2', wing_2_inputs + 0)
    model_1.add(WakeCoords(
        surface_names=surface_names,
        surface_shapes=surface_shapes,
        nt=nt,
        delta_t=1,
    ),
                name='WakeCoords')

    sim = Simulator(model_1)
    # sim.prob.set_val('wing_1', val=wing_1_mesh)
    # sim.prob.set_val('wing_2', val=wing_2_mesh)
    # sim.visualize_implementation()
    sim.run()
