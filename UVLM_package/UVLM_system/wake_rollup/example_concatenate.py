from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import shape, size


class Concatenate(Model):
    """
    A = [a,b
         c,d]
    """
    def initialize(self):

        self.parameters.declare('tenser_names', types=list)
        self.parameters.declare('tenser_shapes', types=list)

        self.parameters.declare('block_matrix_name', types=str)
        self.parameters.declare('block_matrix_structure', types=tuple)
        self.parameters.declare('block_matrix_shape', types=tuple)

    def define(self):
        tenser_names = self.parameters['tenser_names']
        tenser_shapes = self.parameters['tenser_shapes']
        block_matrix_structure = self.parameters['block_matrix_structure']
        block_matrix_shape = self.parameters['block_matrix_shape']
        block_matrix_name = self.parameters['block_matrix_name']

        block_matrix = self.create_output(block_matrix_name,
                                          shape=block_matrix_shape)
        start_row = 0
        start_col = 0
        for i in range(block_matrix_structure[0]):
            for j in range(block_matrix_structure[1]):
                var = self.declare_variable(
                    tenser_names[i * block_matrix_structure[1] + j],
                    shape=tenser_shapes[i * block_matrix_structure[1] + j])

                delta_col = var.shape[1]
                delta_row = var.shape[0]
                block_matrix[start_row:start_row + delta_row,
                             start_col:start_col + delta_col] = var
                start_col += delta_col
            start_col = 0
            start_row += delta_row


if __name__ == "__main__":

    model_1 = Model()

    tenser_names = ['a', 'b', 'c', 'd']
    tenser_shapes = [(2, 2), (2, 3), (3, 4), (3, 1)]
    block_matrix_name = 'A'
    block_matrix_structure = (2, 2)
    block_matrix_shape = (2 + 3, 2 + 3)
    np.random.seed(0)

    for i in range(len(tenser_names)):
        var = model_1.create_input(tenser_names[i],
                                   val=np.random.random(tenser_shapes[i]))

    model_1.add(Concatenate(
        tenser_names=tenser_names,
        tenser_shapes=tenser_shapes,
        block_matrix_name=block_matrix_name,
        block_matrix_structure=block_matrix_structure,
        block_matrix_shape=block_matrix_shape,
    ),
                name='Concatenate')
    sim = Simulator(model_1)
    sim.run()

    for i in range(len(tenser_names)):
        print(tenser_names[i], sim[tenser_names[i]].shape,
              sim[tenser_names[i]])

    print(block_matrix_name, sim[block_matrix_name].shape,
          sim[block_matrix_name])
