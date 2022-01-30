import csdl
from csdl_om import Simulator

import numpy as np
from VLM_package.VLM_system.solve_circulations.solve_group import SolveMatrix
from VLM_package.VLM_preprocessing.mesh_preprocessing_comp import MeshPreprocessing
from VLM_package.VLM_preprocessing.compute_wake_coords import WakeCoords

from VLM_package.VLM_system.solve_circulations.seperate_gamma_b import SeperateGammab


class VLMSystemModel(csdl.Model):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('delta_t', default=100)

        # We also passed in parameters to this ODE model in ODEproblem.create_model() in 'run.py' which we can access here.
        # for now, we just make frame_vel an option, bd_vortex_coords, as static parameters
        self.parameters.declare('frame_vel')
        self.parameters.declare('nt', default=2)
        self.parameters.declare('free_wake', default=False)
        self.parameters.declare('temp_fix_option', default=False)

    def define(self):
        # rename parameters
        n = self.parameters['num_nodes']
        nt = self.parameters['nt']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        free_wake = self.parameters['free_wake']

        temp_fix_option = self.parameters['temp_fix_option']

        wake_coords_names = [x + '_wake_coords' for x in surface_names]

        bd_vortex_shapes = surface_shapes
        delta_t = self.parameters['delta_t']
        gamma_b_shape = sum((i[0] - 1) * (i[1] - 1) for i in bd_vortex_shapes)

        frame_vel_val = self.parameters['frame_vel']

        frame_vel = self.create_input('frame_vel', val=frame_vel_val)
        ode_surface_shape = [(n, ) + item for item in surface_shapes]
        v_total_wake_names = [x + '_wake_total_vel' for x in surface_names]
        wake_vortex_pts_shapes = [
            tuple((nt, item[1], 3)) for item in surface_shapes
        ]
        wake_vel_shapes = [(x[0] * x[1], 3) for x in wake_vortex_pts_shapes]

        ode_bd_vortex_shapes = ode_surface_shape

        self.add(MeshPreprocessing(surface_names=surface_names,
                                   surface_shapes=ode_surface_shape),
                 name='meshPreprocessing_comp')
        self.add(WakeCoords(
            surface_names=surface_names,
            surface_shapes=ode_surface_shape,
            nt=nt,
            delta_t=delta_t,
        ),
                 name='WakeCoords_comp')

        self.add(SolveMatrix(nt=nt,
                             surface_names=surface_names,
                             bd_vortex_shapes=bd_vortex_shapes,
                             delta_t=delta_t),
                 name='solve_gamma_b_group')

        gamma_b = self.declare_variable('gamma_b', shape=gamma_b_shape)

        self.add(SeperateGammab(surface_names=surface_names,
                                surface_shapes=surface_shapes),
                 name='seperate_gamma_b')

        m = csdl.Model()
        sum_ny = sum((i[1] - 1) for i in bd_vortex_shapes)
        gamma_w = m.create_output('gamma_w', shape=(n, nt - 1, sum_ny))
        start = 0
        for i in range(len(surface_names)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            delta = ny - 1

            val = np.zeros((n, nt - 1, ny - 1))
            surface_name = surface_names[i]

            surface_gamma_b_name = surface_name + '_gamma_b'

            surface_gamma_b = m.declare_variable(surface_gamma_b_name,
                                                 shape=((nx - 1) * (ny - 1), ))
            surface_gamma_w_name = surface_names[i] + '_gamma_w'
            surface_gamma_w = csdl.expand(
                surface_gamma_b[(nx - 2) * (ny - 1):], (nt - 1, ny - 1),
                'i->ji')
            m.register_output(surface_gamma_w_name, surface_gamma_w)
            gamma_w[:, :, start:start + delta] = csdl.reshape(
                surface_gamma_w, (1, nt - 1, ny - 1))
            start += delta
        self.add(m, name='extract_gamma_w')

        # gamma_b = self.declare_variable('gamma_b', shape=gamma_b_shape)

        # self.add(SeperateGammab(surface_names=surface_names,
        #                         surface_shapes=surface_shapes),
        #          name='seperate_gamma_b')
        # ODE system with surface gamma's
        for i in range(len(surface_names)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            val = np.zeros((n, nt - 1, ny - 1))
            surface_name = surface_names[i]

            surface_gamma_b_name = surface_name + '_gamma_b'

            surface_gamma_b = self.declare_variable(surface_gamma_b_name,
                                                    shape=((nx - 1) *
                                                           (ny - 1), ))


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

    nt = 5
    nx = 20
    ny = 20
    bd_vortex_coords_val = generate_simple_mesh_mesh(nx, ny)
    delta_t = 1

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
        generate_simple_mesh(nx, ny)[-1, :, 1],
        np.ones(nt),
    ) + (delta_t * np.arange(nt) *
         (-frame_vel_val[1])).reshape(-1, 1)).flatten()

    wake_coords_val = np.zeros((1, nt, ny, 3))
    wake_coords_val[:, :, :, 0] = wake_coords_val_x.reshape(nt, ny)
    wake_coords_val[:, :, :, 1] = wake_coords_val_y.reshape(nt, ny)
    wake_coords_val[:, :, :, 2] = wake_coords_val_z.reshape(nt, ny)

    model_1 = csdl.Model()

    frame_vel_val = np.array([-1, 0, -1])
    mesh_val = generate_simple_mesh_mesh(nx, ny).reshape(1, nx, ny, 3)

    bd_vortex_coords = model_1.create_input('wing', val=mesh_val)
    bd_vortex_coords = model_1.create_input('wing_rot_vel',
                                            val=np.zeros(
                                                ((nx - 1) * (ny - 1), 3)))
    TE = [wake_coords_val.reshape(nt, ny, 3)[0, :, :]]

    model_1.add(
        ODESystemModel(surface_names=['wing'],
                       surface_shapes=[(nx, ny, 3)],
                       frame_vel=frame_vel_val,
                       wake_coords=[wake_coords_val],
                       nt=5,
                       delta_t=delta_t), 'ODE_system')
    sim = Simulator(model_1)
    sim.run()
    print(sim['gamma_b'], 'gamma_b')
    # print(sim['b'], 'b')
    # print(sim['M'], sim['M'].shape, 'M')
    # print(sim['gamma_w'], 'gamma_w')
    # csdl.einsum_new_api
    sim.visualize_implementation()
