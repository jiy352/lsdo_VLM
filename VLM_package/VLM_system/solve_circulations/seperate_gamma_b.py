from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy import random
from numpy.core.fromnumeric import shape, size
from numpy.random import gamma


class SeperateGammab(Model):
    """
    seperate the whole solution vector gamma_b
    corresponding to different lifting surfaces

    parameters
    ----------
    gamma_b

    Returns
    -------
    surface_name+'_gamma_b' : csdl array
    """
    def initialize(self):

        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)

    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']

        num_nodes = surface_shapes[0][0]
        gamma_b_shape = sum((i[1] - 1) * (i[2] - 1) for i in surface_shapes)

        # sum of system_shape with all the nx's and ny's
        gamma_b = self.declare_variable('gamma_b',
                                        shape=(num_nodes, gamma_b_shape))

        start = 0
        for i in range(len(surface_shapes)):
            surface_gamma_b_name = surface_names[i] + '_gamma_b'
            surface_shape = surface_shapes[i]

            nx = surface_shape[1]
            ny = surface_shape[2]
            surface_gamma_b = gamma_b[:, start:start + (nx - 1) * (ny - 1)]
            start += (nx - 1) * (ny - 1)
            # print(surface_gamma_b_name, surface_gamma_b.shape)
            self.register_output(surface_gamma_b_name, surface_gamma_b)


if __name__ == "__main__":

    surface_names = ['a', 'b', 'c']
    surface_shapes = [(3, 2, 3), (2, 4, 3), (2, 4, 3)]
    gamma_b_shape = sum((i[0] - 1) * (i[1] - 1) for i in surface_shapes)

    model_1 = Model()
    gamma_b = model_1.declare_variable('gamma_b',
                                       val=np.random.random((gamma_b_shape)))

    model_1.add(SeperateGammab(
        surface_names=surface_names,
        surface_shapes=surface_shapes,
    ),
                name='SeperateGammab')
    sim = Simulator(model_1)
    sim.run()
    print('gamma_b', gamma_b.shape, gamma_b)
    for i in range(len(surface_shapes)):
        surface_gamma_b_name = surface_names[i] + '_gamma_b'

        print(surface_gamma_b_name, sim[surface_gamma_b_name].shape,
              sim[surface_gamma_b_name])
