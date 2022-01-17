from os import name
import csdl
from csdl_om import Simulator
from ozone2.api import NativeSystem
import numpy as np
from UVLM_package.UVLM_system.solve_circulations.solve_group import SolveMatrix
from UVLM_package.UVLM_preprocessing.mesh_preprocessing_comp import MeshPreprocessing

from UVLM_package.UVLM_system.wake_rollup.seperate_gamma_b import SeperateGammab
from UVLM_package.UVLM_outputs.compute_force.compute_outputs_group import Outputs
from UVLM_package.UVLM_preprocessing.generate_simple_mesh import *

#TODO: how to send the parameter into ode system
#TODO: how to set out put that defined by 2 states in the run file


class ProfileOutputSystemModel(csdl.Model):
    # Computes f_profile that takes state as input and outputs array with specified shape
    def initialize(self):

        self.parameters.declare('nt', default=5)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)

        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('frame_vel_val')

        self.parameters.declare('wake_coords', types=list)

        # self.parameters.declare(
        #     'surface_coords', types=list
        # )  #TODO: need to figure out how to get the parameters in outputs
        self.parameters.declare('free_wake', default=True)
        self.parameters.declare('delta_t')

    def define(self):

        n = self.parameters['num_nodes']
        nt = self.parameters['nt']
        delta_t = self.parameters['delta_t']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        # surface_coords = self.parameters['surface_coords']
        bd_vortex_shapes = surface_shapes

        ode_surface_shapes = [(n, ) + item for item in surface_shapes]
        gamma_b_shape = sum((i[0] - 1) * (i[1] - 1) for i in bd_vortex_shapes)

        eval_pts_names = ['eval_pts_coords']
        eval_pts_shapes = [(x[0] - 1, x[1] - 1, 3) for x in bd_vortex_shapes]
        free_wake = self.parameters['free_wake']

        if free_wake == False:
            wake_coords_val = self.parameters['wake_coords']
        frame_vel_val = self.parameters['frame_vel_val']

        wake_coords_names = [x + '_wake_coords' for x in surface_names]

        vor_coord_shapes = surface_shapes
        nx = vor_coord_shapes[0][0]
        ny = vor_coord_shapes[0][1]

        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            surface_shape = surface_shapes[i]

            # add the parameter
            surface_val = generate_simple_mesh(surface_shape[0],
                                               surface_shape[1], 1)
            surface = self.declare_variable(surface_name, val=surface_val)

        # prepossing and solve for gamma_b
        self.add(MeshPreprocessing(surface_names=surface_names,
                                   surface_shapes=ode_surface_shapes),
                 name='meshPreprocessing_comp')

        self.add(SolveMatrix(nt=nt,
                             surface_names=surface_names,
                             bd_vortex_shapes=bd_vortex_shapes,
                             delta_t=delta_t),
                 name='solve_gamma_b_group')

        gamma_b = self.declare_variable('gamma_b', shape=gamma_b_shape)
        frame_vel = self.create_input('frame_vel', val=frame_vel_val)

        self.add(SeperateGammab(surface_names=surface_names,
                                surface_shapes=surface_shapes),
                 name='seperate_gamma_b')

        for i in range(len(surface_names)):
            surface_gamma_w_name = surface_names[i] + '_gamma_w'
            surface_gamma_w_out_name = surface_names[i] + '_gamma_w_out'
            gamma_w = self.create_input(surface_gamma_w_name,
                                        val=np.zeros((n, nt - 1, ny - 1)))
            gamma_w_out = gamma_w + 0
            self.register_output(surface_gamma_w_out_name, gamma_w_out)

            if free_wake == True:
                surface_wake_coords_name = surface_names[i] + '_wake_coords'
                surface_wake_coords_out_name = surface_names[
                    i] + '_wake_coords_out'
                wake_coords = self.create_input(surface_wake_coords_name,
                                                val=np.zeros((n, nt, ny, 3)))
                wake_coords_out = wake_coords + 0
                self.register_output(surface_wake_coords_out_name,
                                     wake_coords_out)
        # print(surface_shapes, 'surface_shapes')

        # compute lift and drag
        sub = Outputs(surface_names=surface_names,
                      surface_shapes=surface_shapes,
                      nt=nt,
                      eval_pts_names=eval_pts_names,
                      eval_pts_shapes=eval_pts_shapes,
                      delta_t=delta_t)
        self.add(sub, name='compute_lift_drag')


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

    # frame_vel_val = np.array([1, 0, 1])
    bd_vortex_coords_val = generate_simple_mesh_mesh(3, 4)
    # bd_vortex_coords_val[:, :, 0] = bd_vortex_coords_val[:, :, 0]
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

    wake_coords_val = np.zeros((nt, ny, 3))
    wake_coords_val[:, :, 0] = wake_coords_val_x.reshape(nt, ny)
    wake_coords_val[:, :, 1] = wake_coords_val_y.reshape(nt, ny)
    wake_coords_val[:, :, 2] = wake_coords_val_z.reshape(nt, ny)

    model_1 = csdl.Model()

    frame_vel_val = np.array([-1, 0, -1])
    mesh_val = generate_simple_mesh_mesh(3, 4).reshape(1, 3, 4, 3)

    # coll_val = np.random.random((4, 5, 3))

    bd_vortex_coords = model_1.create_input('wing', val=mesh_val)
    # wake_coords = model_1.create_input('wing_wake_coords', val=wake_coords_val)
    model_1.add(
        ProfileOutputSystemModel(
            surface_names=['wing'],
            surface_shapes=[(3, 4, 3)],
            frame_vel_val=frame_vel_val,
            wake_coords=[wake_coords_val],
            nt=5,
        ), 'ODE_outputs')

    sim = Simulator(model_1)
    sim.run()
    print(sim['gamma_b'], 'gamma_b')
    print(sim['b'], 'b')
    print(sim['M'], sim['M'].shape, 'M')
    print(sim['gamma_w'], 'gamma_w')

    sim.visualize_implementation()