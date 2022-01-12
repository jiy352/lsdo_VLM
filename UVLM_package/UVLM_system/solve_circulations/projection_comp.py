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
# projected vs wetted s_ref?
# line 153 why cannot put 0.5 outside the ()


class Projection(Model):
    """
    Compute the normal velocities used to solve the 
    tangential bc.

    parameters
    ----------
    velocities[num_vel, 3] : numpy array
        the velocities.

    Returns
    -------
    velocity_projections[num_vel] : numpy array
        The projected norm of the velocities
    """
    def initialize(self):
        self.parameters.declare('input_vel_names', types=list)
        self.parameters.declare('normal_names', types=list)

        self.parameters.declare('output_vel_names', types=str)

        self.parameters.declare('input_vel_shapes', types=list)
        self.parameters.declare('normal_shapes', types=list)

    def define(self):
        input_vel_names = self.parameters['input_vel_names']
        normal_names = self.parameters['normal_names']
        output_vel_names = self.parameters['output_vel_names']

        input_vel_shapes = self.parameters['input_vel_shapes']
        normal_shapes = self.parameters['normal_shapes']

        output_vel_name = output_vel_names
        # there should only be one concatenated output
        # for both kinematic vel and aic

        input_vel_shape_sum = 0

        # project kinematic vel
        if len(input_vel_shapes[0]) == 2:
            output_shape = sum((i[0]) for i in input_vel_shapes)
        # project aic
        elif len(input_vel_shapes[0]) == 3:
            output_shape = (sum((i[0]) for i in input_vel_shapes)) + (sum(
                (i[1]) for i in input_vel_shapes))

        start = 0
        if len(input_vel_names) > 1:
            output_vel = self.create_output(output_vel_name,
                                            shape=output_shape)

            for i in range(len(input_vel_names)):

                # input_names
                input_vel_name = input_vel_names[i]
                normal_name = normal_names[i]

                # input_shapes
                input_vel_shape = input_vel_shapes[i]
                normal_shape = normal_shapes[i]

                # declare_inputs
                input_vel = self.declare_variable(input_vel_name,
                                                  shape=input_vel_shape)

                normals = self.declare_variable(normal_name,
                                                shape=normal_shape)
                # print('normals shape', normals.shape)
                normals_reshaped = csdl.reshape(
                    normals,
                    new_shape=(normals.shape[0] * normals.shape[1], 3))
                # print('finsih reshape')
                # print('input_vel shape', input_vel.shape)
                if len(input_vel_shape) == 2:
                    velocity_projections = csdl.einsum(input_vel,
                                                       normals_reshaped,
                                                       subscripts='ij,ij->i')
                elif len(input_vel_shape) == 3:
                    print(
                        'Implementation error the dim of kinematic vel should be 2'
                    )
                    print(input_vel_name, normal_name, output_vel_name)
                    exit()
                    # velocity_projections = csdl.einsum(input_vel,
                    #                                    normals_reshaped,
                    #                                    subscripts='ijk,ik->ij')
                # print('finsih velocity_projections')
                delta = velocity_projections.shape[0]
                output_vel[start:start + delta] = velocity_projections
                start += delta

        elif len(input_vel_names) == 1:
            input_vel_name = input_vel_names[0]
            input_vel_shape = input_vel_shapes[0]
            # we need to concatenate the normal vectors
            # into a whole vec to project the assembled aic matrix

            normal_concatenated_shape = (sum(
                (i[0] * i[1]) for i in normal_shapes), ) + (3, )

            print('normal_concatenated_shape-----------',
                  normal_concatenated_shape)

            normal_concatenated = self.create_output(
                'normal_concatenated' + '_' + output_vel_name,
                shape=normal_concatenated_shape)

            for i in range(len(normal_names)):

                # input_names

                normal_name = normal_names[i]

                # input_shapes

                normal_shape = normal_shapes[i]

                # declare_inputs
                input_vel = self.declare_variable(input_vel_name,
                                                  shape=input_vel_shape)

                normals = self.declare_variable(normal_name,
                                                shape=normal_shape)
                # print('normals shape', normals.shape)

                normals_reshaped = csdl.reshape(
                    normals,
                    new_shape=(normals.shape[0] * normals.shape[1], 3))

                delta = normals_reshaped.shape[0]
                # print('find bug0-----------')
                normal_concatenated[start:start + delta, :] = normals_reshaped
                # print('finsih reshape')
                # print('input_vel shape', input_vel.shape)
                start += delta
            # print('find bug1-----------')
            if len(input_vel_shape) == 2:
                print('warning: the dim of aic should be 3')
                print(input_vel_name, normal_name, output_vel_name)
                velocity_projections = csdl.einsum(input_vel,
                                                   normals_reshaped,
                                                   subscripts='ij,ij->i')
                self.register_output(output_vel_name, velocity_projections)
            elif len(input_vel_shape) == 3:
                velocity_projections = csdl.einsum(input_vel,
                                                   normal_concatenated,
                                                   subscripts='ijk,ik->ij')
                self.register_output(output_vel_name, velocity_projections)
                # print('find bug2-----------')
                print('find bug2--------------------', output_vel_name,
                      velocity_projections.shape)
                print('input shapes----------', input_vel_name,
                      input_vel.shape, normal_concatenated.shape)


if __name__ == "__main__":

    def generate_simple_mesh(nx, ny, nt=None):
        if nt == None:
            mesh = np.zeros((nx, ny, 3))
            mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))
            mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
            mesh[:, :, 2] = 0.
        else:
            mesh = np.zeros((nt, nx, ny, 3))
            for i in range(nt):
                mesh[i, :, :, 0] = np.outer(np.arange(nx), np.ones(ny))
                mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
                mesh[i, :, :, 2] = 0.
        return mesh

    input_vel_names = ['i1', 'i2']
    normals_names = ['n1', 'n2']
    output_vel_names = ['o1', 'o2']

    # input_vel_shapes = [(2, 3), (4, 3)]
    input_vel_shapes = [(2, 4, 3), (4, 5, 3)]
    normal_shapes = [(2, 3), (4, 3)]

    model_1 = Model()
    # i1_val = np.random.random((2, 3))
    # i2_val = np.random.random((4, 3))

    i1_val = np.random.random((2, 4, 3))
    i2_val = np.random.random((4, 5, 3))
    i1 = model_1.create_input('i1', val=i1_val)
    i2 = model_1.create_input('i2', val=i2_val)
    model_1.add(Projection(
        input_vel_names=input_vel_names,
        normal_names=normals_names,
        output_vel_names=output_vel_names,
        input_vel_shapes=input_vel_shapes,
        normal_shapes=normal_shapes,
    ),
                name='Projection')
    sim = Simulator(model_1)
    # sim.visualize_implementation()
    sim.run()
