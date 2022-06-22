from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size
import random
from VLM_package.VLM_system.solve_circulations.utils.einsum_kij_kij_ki import EinsumKijKijKi
from VLM_package.VLM_system.solve_circulations.utils.einsum_lijk_lik_lij import EinsumLijkLikLij


class Projection(Model):
    """
    Compute the normal velocities used to solve the 
    tangential bc.

    parameters
    ----------
    velocities[num_nodes,num_vel, 3] : numpy array
        the velocities. num_vel = (nx-1)*(ny-1)
    normals[num_nodes,nx-1,ny-1, 3] : numpy array
        the normals.
    Returns
    -------
    For kinematic velocities:
    velocity_projections[num_nodes,num_vel] : numpy array
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

        num_nodes = normal_shapes[0][0]
        # print('compute_normal line 41 input_vel_names', input_vel_names)
        # print('compute_normal line 42 input_vel_shapes', input_vel_shapes)
        # print('compute_normal line 43 normal_shapes', normal_shapes)

        output_vel_name = output_vel_names
        # there should only be one concatenated output
        # for both kinematic vel and aic

        input_vel_shape_sum = 0

        # project kinematic vel
        if len(input_vel_shapes[0]) == 3:
            output_shape = (num_nodes, sum((i[1]) for i in input_vel_shapes))
        # project aic
        elif len(input_vel_shapes[0]) == 4:
            output_shape = (num_nodes, +((sum(
                (i[1]) for i in input_vel_shapes)) + (sum(
                    (i[2]) for i in input_vel_shapes))))

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
                    new_shape=(num_nodes, normals.shape[1] * normals.shape[2],
                               3))

                if len(input_vel_shape) == 3:
                    velocity_projections = csdl.einsum(
                        input_vel,
                        normals_reshaped,
                        subscripts='ijk,ijk->ij',
                        partial_format='sparse',
                    )
                elif len(input_vel_shape) == 4:
                    print(
                        'Implementation error the dim of kinematic vel should be 3'
                    )
                    print(input_vel_name, normal_name, output_vel_name)
                    exit()

                delta = velocity_projections.shape[1]
                # print('projection comp output_vel shape', output_vel.shape)
                # print('projection comp velocity_projections shape',
                #       velocity_projections.shape)
                # print('projection comp start start + delta', start,
                #       start + delta)
                output_vel[:, start:start + delta] = velocity_projections
                start += delta
            # start = 0

        elif len(input_vel_names) == 1:
            input_vel_name = input_vel_names[0]
            input_vel_shape = input_vel_shapes[0]
            # we need to concatenate the normal vectors
            # into a whole vec to project the assembled aic matrix

            normal_concatenated_shape = (num_nodes, ) + (sum(
                (i[1] * i[2]) for i in normal_shapes), ) + (3, )

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
                    new_shape=(num_nodes, normals.shape[1] * normals.shape[2],
                               3))
                if len(input_vel_shape) == 3:
                    self.register_output(normal_name + '_reshaped',
                                         normals_reshaped)
                    velocity_projections = csdl.custom(
                        input_vel,
                        normals_reshaped,
                        op=EinsumKijKijKi(in_name_1=input_vel_name,
                                          in_name_2=normal_name + '_reshaped',
                                          in_shape=input_vel.shape,
                                          out_name=output_vel_name))

                    self.register_output(output_vel_name, velocity_projections)
                delta = normals_reshaped.shape[1]
                normal_concatenated[:,
                                    start:start + delta, :] = normals_reshaped

                start += delta
            # if len(input_vel_shape) == 3:
            #     # print('warning: the dim of aic should be 4')
            #     # print(input_vel_name, normal_name, output_vel_name)

            #     # velocity_projections = csdl.einsum(
            #     #     input_vel,
            #     #     normals_reshaped,
            #     #     subscripts='kij,kij->ki',
            #     #     partial_format='sparse',
            #     # )
            #     self.register_output(normal_name + '_reshaped',
            #                          normals_reshaped)

            #     velocity_projections = csdl.custom(
            #         input_vel,
            #         normals_reshaped,
            #         op=EinsumKijKijKi(in_name_1=input_vel_name,
            #                           in_name_2=normal_name + '_reshaped',
            #                           in_shape=input_vel.shape,
            #                           out_name=output_vel_name))

            #     self.register_output(output_vel_name, velocity_projections)
            if len(input_vel_shape) == 4:
                # velocity_projections = csdl.einsum(
                #     input_vel,
                #     normal_concatenated,
                #     subscripts='lijk,lik->lij',
                #     partial_format='sparse',
                # )
                # self.register_output(normal_name + '_cat', normal_concatenated)

                velocity_projections = csdl.custom(
                    input_vel,
                    normal_concatenated,
                    op=EinsumLijkLikLij(in_name_1=input_vel_name,
                                        in_name_2='normal_concatenated' + '_' +
                                        output_vel_name,
                                        in_shape=input_vel.shape,
                                        out_name=output_vel_name))

                self.register_output(output_vel_name, velocity_projections)


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

    '''1. testing project kinematic vel'''
    input_vel_names = ['i1']
    normals_names = ['n1']
    output_vel_names = 'o1'
    num_nodes = 2
    # input_vel_shapes = [(2, 3), (4, 3)]
    input_vel_shapes = [(num_nodes, 3, 3)]
    normal_shapes = [(num_nodes, 1, 3, 3)]

    model_1 = Model()
    # i1_val = np.random.random((2, 3))
    # i2_val = np.random.random((4, 3))

    i1_val = np.random.random(input_vel_shapes[0])
    i1 = model_1.create_input('i1', val=i1_val)
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
    '''2. testing project aic'''
