import csdl
from csdl_om import Simulator
from ozone2.api import NativeSystem
import numpy as np
# from solve_group_simple_wing import SolveMatrix
from UVLM_package.UVLM_system.solve_circulations.solve_group_new import SolveMatrix
from UVLM_package.UVLM_preprocessing.mesh_preprocessing_comp import MeshPreprocessing


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

    def define(self):
        # Input: state
        n = self.parameters['num_nodes']
        nt = self.parameters['nt']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']

        wake_coords_names = [x + '_wake_coords' for x in surface_names]

        bd_vortex_shapes = surface_shapes
        delta_t = self.parameters['delta_t']

        frame_vel_val = self.parameters['frame_vel']
        wake_coords_val = self.parameters['wake_coords']

        frame_vel = self.create_input('frame_vel', val=frame_vel_val)
        ode_surface_shape = [(n, ) + item for item in surface_shapes]

        ode_bd_vortex_shapes = ode_surface_shape

        for i in range(len(surface_names)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            print('nx, ny', bd_vortex_shapes[i], nx, ny)
            # NOTE changed here from create input
            self.declare_variable(surface_names[i], shape=ode_surface_shape[i])
            wake_coords = self.create_input(wake_coords_names[i],
                                            val=wake_coords_val[i])

        self.add(MeshPreprocessing(surface_names=surface_names,
                                   surface_shapes=ode_surface_shape),
                 name='meshPreprocessing_comp')

        self.add(SolveMatrix(nt=nt,
                             surface_names=surface_names,
                             bd_vortex_shapes=bd_vortex_shapes),
                 name='solve_gamma_b_group')

        delta_t = self.parameters['delta_t']
        gamma_b = self.declare_variable('gamma_b',
                                        shape=((nx - 1) * (ny - 1), ))

        val = np.zeros((n, nt - 1, ny - 1))
        gamma_w = self.create_input('gamma_w', val=val)
        dgammaw_dt = self.create_output('dgammaw_dt',
                                        shape=(n, nt - 1, ny - 1))

        for i in range(n):
            gamma_b_last = csdl.reshape(gamma_b[(nx - 2) * (ny - 1):],
                                        new_shape=(1, 1, ny - 1))

            dgammaw_dt[i, 0, :] = gamma_b_last - gamma_w[i, 0, :]
            dgammaw_dt[i, 1:, :] = (gamma_w[i, :(gamma_w.shape[1] - 1), :] -
                                    gamma_w[i, 1:, :]) / delta_t


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

    wake_coords_val = np.zeros((nt, ny, 3))
    wake_coords_val[:, :, 0] = wake_coords_val_x.reshape(nt, ny)
    wake_coords_val[:, :, 1] = wake_coords_val_y.reshape(nt, ny)
    wake_coords_val[:, :, 2] = wake_coords_val_z.reshape(nt, ny)

    model_1 = csdl.Model()

    frame_vel_val = np.array([-1, 0, -1])
    mesh_val = generate_simple_mesh_mesh(3, 4).reshape(1, 3, 4, 3)

    bd_vortex_coords = model_1.create_input('wing', val=mesh_val)
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
    print(sim['gamma_w'], 'gamma_w')

    sim.visualize_implementation()
