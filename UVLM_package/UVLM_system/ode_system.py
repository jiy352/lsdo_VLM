from os import times_result
from random import random
import csdl
from csdl_om import Simulator
from numpy.core.fromnumeric import shape
from ozone2.api import NativeSystem
import numpy as np
# from solve_group_simple_wing import SolveMatrix
from UVLM_package.UVLM_system.solve_circulations.solve_group import SolveMatrix
from UVLM_package.UVLM_preprocessing.mesh_preprocessing_comp import MeshPreprocessing

from UVLM_package.UVLM_system.wake_rollup.seperate_gamma_b import SeperateGammab
from UVLM_package.UVLM_system.wake_rollup.wake_total_vel_group import WakeTotalVel
from random import randrange


class ODESystemModel(csdl.Model):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('delta_t', default=1)

        # We also passed in parameters to this ODE model in ODEproblem.create_model() in 'run.py' which we can access here.
        # for now, we just make frame_vel an option, bd_vortex_coords, as static parameters
        self.parameters.declare('frame_vel')
        self.parameters.declare('wake_coords', types=list)
        self.parameters.declare('nt', types=int)
        self.parameters.declare('free_wake', default=True)

    def define(self):
        # rename parameters
        n = self.parameters['num_nodes']
        nt = self.parameters['nt']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        free_wake = self.parameters['free_wake']

        wake_coords_names = [x + '_wake_coords' for x in surface_names]

        bd_vortex_shapes = surface_shapes
        delta_t = self.parameters['delta_t']
        gamma_b_shape = sum((i[0] - 1) * (i[1] - 1) for i in bd_vortex_shapes)

        frame_vel_val = self.parameters['frame_vel']
        wake_coords_val = self.parameters['wake_coords']

        frame_vel = self.create_input('frame_vel', val=frame_vel_val)
        ode_surface_shape = [(n, ) + item for item in surface_shapes]
        v_total_wake_names = [x + '_wake_total_vel' for x in surface_names]
        wake_vortex_pts_shapes = [
            tuple((nt, item[1], 3)) for item in surface_shapes
        ]
        wake_vel_shapes = [(x[0] * x[1], 3) for x in wake_vortex_pts_shapes]

        ode_bd_vortex_shapes = ode_surface_shape

        if free_wake == True:
            vel_coeff = self.create_input('vel_coeff', shape=(n, nt - 1))

        for i in range(len(surface_names)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            print('nx, ny', bd_vortex_shapes[i], nx, ny)
            # NOTE changed here from create input
            self.declare_variable(surface_names[i], shape=ode_surface_shape[i])
            # ! TODO ! fix here for free wake option
            if free_wake == False:
                wake_coords = self.declare_variable(wake_coords_names[i],
                                                    val=wake_coords_val[i])

        self.add(MeshPreprocessing(surface_names=surface_names,
                                   surface_shapes=ode_surface_shape),
                 name='meshPreprocessing_comp')

        self.add(SolveMatrix(nt=nt,
                             surface_names=surface_names,
                             bd_vortex_shapes=bd_vortex_shapes),
                 name='solve_gamma_b_group')

        delta_t = self.parameters['delta_t']
        gamma_b = self.declare_variable('gamma_b', shape=gamma_b_shape)

        self.add(SeperateGammab(surface_names=surface_names,
                                surface_shapes=surface_shapes),
                 name='seperate_gamma_b')
        # ODE system with surface gamma's
        for i in range(len(surface_names)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            val = np.zeros((n, nt - 1, ny - 1))
            surface_name = surface_names[i]

            surface_gamma_w_name = surface_name + '_gamma_w'
            surface_dgammaw_dt_name = surface_name + '_dgammaw_dt'
            surface_gamma_b_name = surface_name + '_gamma_b'

            surface_gamma_w = self.declare_variable(surface_gamma_w_name,
                                                    shape=val.shape)
            surface_gamma_b = self.declare_variable(surface_gamma_b_name,
                                                    shape=((nx - 1) *
                                                           (ny - 1), ))
            surface_dgammaw_dt = self.create_output(surface_dgammaw_dt_name,
                                                    shape=(n, nt - 1, ny - 1))
            if free_wake == True:

                self.add(
                    WakeTotalVel(surface_names=surface_names,
                                 surface_shapes=surface_shapes,
                                 nt=nt), 'Wake_total_vel_group')
                surface_wake_coords_name = surface_name + '_wake_coords'
                surface_dwake_coords_dt_name = surface_name + '_dwake_coords_dt'

                surface_wake_coords = self.create_input(
                    surface_wake_coords_name, shape=(n, nt, ny, 3))

                print('surface_wake_coords----------------',
                      surface_wake_coords.shape)
                surface_dwake_coords_dt = self.create_output(
                    surface_dwake_coords_dt_name, shape=(n, nt, ny, 3))

            for j in range(n):
                gamma_b_last = csdl.reshape(surface_gamma_b[(nx - 2) *
                                                            (ny - 1):],
                                            new_shape=(1, 1, ny - 1))

                surface_dgammaw_dt[j,
                                   0, :] = (gamma_b_last -
                                            surface_gamma_w[j, 0, :]) / delta_t
                surface_dgammaw_dt[j, 1:, :] = (
                    surface_gamma_w[j, :(surface_gamma_w.shape[1] - 1), :] -
                    surface_gamma_w[j, 1:, :]) / delta_t
                if free_wake == True:
                    # here, we compute the wake coords
                    # The zero th column will always be the T.E. won't change forever
                    # when suppose nt=4 0,1,2,3

                    # t=0       [TE,              TE,                 TE,                TE]
                    # t = 1,    [TE,              TE+v_ind(TE,w+bd),  TE,                TE] -> bracket 0-1
                    # c11 = TE+v_ind(TE,w+bd)

                    # t = 2,    [TE,              TE+v_ind(t=1, bracket 0),  c11+v_ind(t=1, bracket 1),   TE] ->  bracket 0-1-2
                    # c21 =  TE+v_ind(t=1, bracket 0)
                    # c22 =  c11+v_ind(t=1, bracket 1)

                    # t = 3,    [TE,              TE+v_ind(t=2, bracket 0),  c21+vind(t=2, bracket 1), c22+vind(t=2, bracket 2)] -> bracket 0-1-2-3
                    # Then, the shedding is

                    zeros = self.create_input('zeros',
                                              val=np.zeros((1, 1, ny, 3)))

                    surface_dwake_coords_dt[0, 0, :, :] = zeros

                    frame_vel_expand = -csdl.expand(
                        frame_vel, shape=(1, nt - 1, ny, 3), indices='i->jkli')

                    wake_total_vel_final = csdl.einsum(
                        vel_coeff,  #3,3
                        frame_vel_expand,
                        subscripts='ij,ijkl->ijkl')

                    print('shapes in ode_system', )
                    print('wake_total_vel_final', wake_total_vel_final.shape)
                    print(
                        'shapes in ode_system',
                        surface_wake_coords[j, :(surface_wake_coords.shape[1] -
                                                 1), :, :].shape)
                    print('shapes in ode_system',
                          surface_wake_coords[j, 1:, :, :].shape)

                    # self.add(
                    #     WakeTotalVel(surface_names=surface_names,
                    #                  surface_shapes=surface_shapes,
                    #                  nt=nt), 'Wake_total_vel_group')

                    # nt*ny,3

                    wake_total_vel = self.declare_variable(
                        v_total_wake_names[i], val=np.zeros((nt, ny, 3)))

                    wake_total_vel_reshaped = csdl.reshape(
                        wake_total_vel, (1, nt, ny, 3))

                    surface_dwake_coords_dt[0, 1:, :, :] = (
                        surface_wake_coords[j, :(surface_wake_coords.shape[1] -
                                                 1), :, :] -
                        surface_wake_coords[j, 1:, :, :]
                    ) / delta_t + wake_total_vel_reshaped[:, :(
                        surface_wake_coords.shape[1] - 1), :, :]
                    # + wake_total_vel_reshaped[:, :(
                    # surface_wake_coords.shape[1] - 1), :, :]

                    #  + frame_vel_expand

                    # print(
                    #     'shapes to linear comb', surface_wake_coords.shape,
                    # surface_wake_coords[i, :(surface_gamma_w.shape[1] -
                    #                          1), :, :].shape,
                    # surface_wake_coords[i, 1:, :, :].shape)
                    # print('vel_coeff shape-----------', vel_coeff.shape)
                    # print('wake_total_vel_reshaped shape-----------',
                    #       wake_total_vel_reshaped.shape)
                    # wake_total_vel_reshaped = csdl.reshape(
                    #     wake_total_vel, (1, nt, ny, 3))

                    # test if the fix wake coords influence for free wake

                    # wake_total_vel_final = csdl.einsum(
                    #     vel_coeff,
                    #     wake_total_vel_reshaped,
                    #     subscripts='ij,ijkl->ijkl') / delta_t
                    # surface_dwake_coords_dt = wake_total_vel_final

                    # frame_vel_expand = -csdl.expand(
                    #     frame_vel, shape=(1, nt, ny, 3), indices='i->jkli')

                    # wake_total_vel_final = csdl.einsum(
                    #     vel_coeff,
                    #     frame_vel_expand,
                    #     subscripts='ij,ijkl->ijkl') / delta_t

                    # surface_dwake_coords_dt = wake_total_vel_final
                    # self.register_output(surface_dwake_coords_dt_name,
                    #                      surface_dwake_coords_dt)
        print('finish ode system----------------------------')


if __name__ == "__main__":

    def generate_simple_mesh(nx, ny, nt=None):
        if nt == None:
            mesh = np.zeros((nx, ny, 3))
            mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny)) + 0.25
            mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
            mesh[:, :, 2] = 0.
        else:
            mesh = np.zeros((nt, nx, ny, 3))
            for i in range(nt):
                mesh[i, :, :, 0] = np.outer(np.arange(nx), np.ones(ny)) + 0.25
                mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
                mesh[i, :, :, 2] = 0.
        return mesh

    def generate_simple_mesh_mesh(nx, ny, nt=None):
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

    bd_vortex_coords_val = generate_simple_mesh_mesh(3, 4)
    delta_t = 1
    nt = 5
    nx = 3
    ny = 4

    frame_vel_val = np.array([-1, 0, -1])
    wake_coords_val_x = np.einsum(
        'i,j->ij',
        (2.25 + delta_t * np.arange(nt) * (-frame_vel_val[0])),
        np.ones(ny),
    ).flatten()

    wake_coords_val_z = np.einsum(
        'i,j->ij',
        (0 + delta_t * np.arange(nt) * (-frame_vel_val[2])),
        np.ones(ny),
    ).flatten()

    wake_coords_val_y = (np.einsum(
        'i,j->ji',
        generate_simple_mesh(3, 4)[-1, :, 1],
        np.ones(nt),
    ) + (delta_t * np.arange(nt) *
         (-frame_vel_val[1])).reshape(-1, 1)).flatten()

    wake_coords_val = np.zeros((1, nt, ny, 3))
    wake_coords_val[:, :, :, 0] = wake_coords_val_x.reshape(nt, ny)
    wake_coords_val[:, :, :, 1] = wake_coords_val_y.reshape(nt, ny)
    wake_coords_val[:, :, :, 2] = wake_coords_val_z.reshape(nt, ny)

    model_1 = csdl.Model()

    frame_vel_val = np.array([-1, 0, -1])
    mesh_val = generate_simple_mesh_mesh(3, 4).reshape(1, 3, 4, 3)
    val = np.array([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]])
    bd_vortex_coords = model_1.create_input('wing_gamma_w', val=val)

    bd_vortex_coords = model_1.create_input('wing', val=mesh_val)
    bd_vortex_coords = model_1.create_input('wing_rot_vel',
                                            val=np.zeros(
                                                ((nx - 1) * (ny - 1), 3)))
    TE = [wake_coords_val.reshape(nt, ny, 3)[0, :, :]]
    # bd_vortex_coords = model_1.create_input('wing_wake_coords',
    #                                         val=np.einsum(
    #                                             'i,jk->ijk',
    #                                             np.ones(nt),
    #                                             TE[0],
    #                                         ).reshape(1, nt, ny, 3))

    model_1.add(
        ODESystemModel(
            surface_names=['wing'],
            surface_shapes=[(3, 4, 3)],
            frame_vel=frame_vel_val,
            wake_coords=[wake_coords_val],
            nt=5,
        ), 'ODE_system')
    sim = Simulator(model_1)
    sim.run()
    print(sim['gamma_b'], 'gamma_b')
    print(sim['b'], 'b')
    print(sim['M'], sim['M'].shape, 'M')
    # print(sim['gamma_w'], 'gamma_w')

    sim.visualize_implementation()
