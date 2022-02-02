from turtle import shape
from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size


class BoundVec(Model):
    """
    compute bound vectors given the vortex coords at 1/4 chord
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)

    def define(self):
        # add_input
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']

        bound_vecs_names = [x + '_bound_vecs' for x in surface_names]

        for i in range(len(surface_names)):
            nx = surface_shapes[i][0]
            ny = surface_shapes[i][1]

            mesh = self.declare_variable(surface_names[i],
                                         shape=(1, ) + surface_shapes[i])
            bound_vecs = csdl.reshape(
                (0.75 * mesh[:, 0:-1, 0:-1, :] + 0.25 * mesh[:, 1:, 0:-1, :] +
                 -0.75 * mesh[:, 0:-1, 1:, :] + -0.25 * mesh[:, 1:, 1:, :]),
                new_shape=((nx - 1) * (ny - 1), 3))

            self.register_output(bound_vecs_names[i], bound_vecs)

        system_size = 0
        for i in range(len(surface_names)):
            nx = surface_shapes[i][0]
            ny = surface_shapes[i][1]
            system_size += (nx - 1) * (ny - 1)

        combine_bd_vec = Model()

        # combine bd_vecs
        bd_vec_all = combine_bd_vec.create_output('bd_vec',
                                                  shape=(system_size, 3))
        start = 0
        for i in range(len(surface_names)):
            # print(i)
            nx = surface_shapes[i][0]
            ny = surface_shapes[i][1]
            bound_vecs = combine_bd_vec.declare_variable(bound_vecs_names[i],
                                                         shape=((nx - 1) *
                                                                (ny - 1), 3))
            delta = (nx - 1) * (ny - 1)
            # print('start', start, 'delta', delta)
            bd_vec_all[start:start + delta, :] = bound_vecs
            start += delta
        self.add(combine_bd_vec, 'combine_bd_vec')


if __name__ == "__main__":
    nt = 5
    nx = 3
    ny = 4

    def generate_simple_mesh(nx, ny):
        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0] = np.outer(np.arange(nx),
                                 np.ones(ny))  #+ np.random.random(
        #  (nx, ny))
        mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
        mesh[:, :, 2] = 0.
        return mesh

    surface_names = ['wing_1', 'wing_2']
    surface_shapes = [(nx, ny, 3), (nx + 1, ny + 1, 3)]
    model_1 = Model()

    wing_1_mesh = generate_simple_mesh(nx, ny)
    wing_2_mesh = generate_simple_mesh(nx + 1, ny + 1)

    f = model_1.create_input('frame_vel', val=np.array([-1, 0, -1]))

    wing_1_inputs = model_1.create_input('wing_1',
                                         val=wing_1_mesh.reshape(1, nx, ny, 3))
    wing_2_inputs = model_1.create_input('wing_2',
                                         val=wing_2_mesh.reshape(
                                             1, nx + 1, ny + 1, 3))
    # model_1.register_output('wing_1', wing_1_inputs + 0)
    # model_1.register_output('wing_2', wing_2_inputs + 0)
    model_1.add(BoundVec(
        surface_names=surface_names,
        surface_shapes=surface_shapes,
    ),
                name='BoundVec')

    sim = Simulator(model_1)
    # sim.prob.set_val('wing_1', val=wing_1_mesh)
    # sim.prob.set_val('wing_2', val=wing_2_mesh)
    # sim.visualize_implementation()
    sim.run()
