import csdl
from csdl_om import Simulator
from ozone2.api import NativeSystem
import numpy as np
from UVLM_package.UVLM_system.solve_circulations.solve_group_new import SolveMatrix
from UVLM_package.UVLM_preprocessing.mesh_preprocessing_comp import MeshPreprocessing


class ProfileOutputSystemModel(csdl.Model):
    # Computes f_profile that takes state as input and outputs array with specified shape
    def initialize(self):

        self.parameters.declare('nt', default=5)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)

        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('frame_vel_val')

        self.parameters.declare('wake_coords', types=list)

        self.parameters.declare(
            'surface_coords', types=list
        )  #TODO: need to figure out how to get the parameters in outputs

    def define(self):
        n = self.parameters['num_nodes']
        nt = self.parameters['nt']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        surface_coords = self.parameters['surface_coords']

        wake_coords_val = self.parameters['wake_coords']
        frame_vel_val = self.parameters['frame_vel_val']

        wake_coords_names = [x + '_wake_coords' for x in surface_names]

        vor_coord_shapes = surface_shapes
        nx = vor_coord_shapes[0][0]
        ny = vor_coord_shapes[0][1]

        delta_t = 1.

        gamma_w = self.create_input('gamma_w',
                                    val=np.zeros((n, nt - 1, ny - 1)))

        gamma_w_all = gamma_w + 0

        # TODO fix this part for mls

        ode_surface_shape = [(n, ) + item for item in surface_shapes]

        ode_bd_vortex_shapes = ode_surface_shape

        for i in range(len(surface_names)):

            # NOTE changed here from create input
            self.create_input(surface_names[i], val=surface_coords[i])

        wake_coords = self.create_input(wake_coords_names[0],
                                        val=wake_coords_val[0])

        self.add(MeshPreprocessing(surface_names=surface_names,
                                   surface_shapes=ode_surface_shape),
                 name='meshPreprocessing_comp')
        self.add(SolveMatrix(nt=nt,
                             surface_names=surface_names,
                             bd_vortex_shapes=vor_coord_shapes),
                 name='solve_gamma_b_outputs')

        gamma_b = self.declare_variable('gamma_b',
                                        shape=((nx - 1) * (ny - 1), ))
        frame_vel = self.create_input('frame_vel', val=frame_vel_val)
        aic_bd = self.declare_variable('aic_bd',
                                       shape=(
                                           (nx - 1) * (ny - 1),
                                           (nx - 1) * (ny - 1),
                                           3,
                                       ))
        aic_M = self.declare_variable('aic_M',
                                      shape=(
                                          (nx - 1) * (ny - 1),
                                          (nt - 1) * (ny - 1),
                                          3,
                                      ))
        v_induced_bd = csdl.einsum(
            aic_bd, gamma_b,
            subscripts='ijk,j->ik')  # bd_vel induced by the bound
        v_induced_M = csdl.einsum(
            aic_M,
            csdl.reshape(gamma_w, new_shape=((nt - 1) * (ny - 1))),
            subscripts='ijk,j->ik')  # bd_vel induced by the wake
        v_induced = v_induced_bd + v_induced_M
        frame_vel_expand = csdl.expand(frame_vel,
                                       shape=((nx - 1) * (ny - 1), 3),
                                       indices='j->ij')
        rho = 1.225

        delta_p = rho * (
            (csdl.pnorm(-frame_vel_expand + v_induced, axis=(1, )))**2 -
            (csdl.pnorm(-frame_vel_expand, axis=(1, ))**2)) / 2

        C_l = -csdl.sum(csdl.reshape(delta_p,
                                     new_shape=(n, (nx - 1) * (ny - 1))),
                        axes=(1, )) * np.cos(
                            np.arctan(frame_vel_val[2] / frame_vel_val[0])) / (
                                0.5 * rho * 6) / (csdl.pnorm(-frame_vel)**2)

        # TODO: fix this hard coding using a csdl model
        self.register_output(
            'delta_p', csdl.reshape(delta_p,
                                    new_shape=(n, (nx - 1) * (ny - 1))))

        self.register_output(
            'v_induced_bd',
            csdl.reshape(v_induced_bd, new_shape=(n, (nx - 1) * (ny - 1), 3)))
        self.register_output(
            'v_induced_w',
            csdl.reshape(v_induced_M, new_shape=(n, (nx - 1) * (ny - 1), 3)))
        self.register_output('gamma_w_all', gamma_w_all)
        self.register_output('C_l', csdl.reshape(C_l, new_shape=(n, 1)))


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