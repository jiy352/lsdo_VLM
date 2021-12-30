import csdl
from csdl_om import Simulator
from ozone2.api import NativeSystem
import numpy as np
from UVLM_package.UVLM_system.solve_circulations.solve_group_new import SolveMatrix


class ProfileOutputSystem(NativeSystem):
    # Computes f_profile that takes state as input and outputs array with specified shape
    def initialize(self):
        self.parameters.declare('nx')
        self.options.declare('ny')
        self.options.declare('delta_t')
        self.options.declare('nt')

    def setup(self):
        n = 1
        nt = self.parameters['nt']
        ny = 4
        delta_t = 1.
        self.add_input('gamma_w', shape=(n, nt - 1, ny - 1))
        self.add_output('gamma_w_all', shape=(n, nt - 1, ny - 1))
        print('---------------ProfileOutputSystem-----setup----finish')

    def compute(self, inputs, outputs):
        print('---------------ProfileOutputSystem-----compute----enter')

        # n = self.num_nodes
        # print(inputs['y'].shape)
        outputs['gamma_w_all'] = inputs['gamma_w']
        print('---------------ProfileOutputSystem-----compute----end')

    def compute_partials(self, inputs, partials):
        n = 1
        nt = 5
        ny = 4

        print(
            '---------------ProfileOutputSystem-----compute_partials----before'
        )

        partials['gamma_w_all']['gamma_w'] = np.eye(n * (nt - 1) * (ny - 1),
                                                    n * (nt - 1) * (ny - 1))
        print(
            '---------------ProfileOutputSystem-----compute_partials----after')


class ProfileOutputSystemModel(csdl.Model):
    # Computes f_profile that takes state as input and outputs array with specified shape
    def initialize(self):

        self.parameters.declare('nt', default=5)
        self.parameters.declare('vortex_coords_shapes', types=list)
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('frame_vel_val')

    def define(self):
        n = self.parameters['num_nodes']
        nt = self.parameters['nt']
        vor_coord_shapes = self.parameters['vortex_coords_shapes']
        nx = vor_coord_shapes[0][0]
        ny = vor_coord_shapes[0][1]

        # nx = 3
        # ny = 4
        delta_t = 1.
        # TODO: figure out how to set parameter for outputs
        bd_vortex_coords_val = self._generate_simple_mesh(nx, ny)
        frame_vel_val = self.parameters['frame_vel_val']
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
            self._generate_simple_mesh(nx, ny)[-1, :, 1],
            np.ones(nt),
        ) + (delta_t * np.arange(nt) *
             (-frame_vel_val[1])).reshape(-1, 1)).flatten()
        wake_coords_val = np.zeros((nt, ny, 3))
        wake_coords_val[:, :, 0] = wake_coords_val_x.reshape(nt, ny)
        wake_coords_val[:, :, 1] = wake_coords_val_y.reshape(nt, ny)
        wake_coords_val[:, :, 2] = wake_coords_val_z.reshape(nt, ny)

        # gamma_w = self.create_input('gamma_w', shape=(n, nt - 1, ny - 1))
        gamma_w = self.create_input('gamma_w',
                                    val=np.zeros((n, nt - 1, ny - 1)))

        gamma_w_all = gamma_w + 0

        model = csdl.Model()
        bd_vortex_coord_val = bd_vortex_coords_val
        coll_pts_val = 0.25 * (bd_vortex_coord_val[0:nx-1, 0:ny-1, :] +\
                                                bd_vortex_coord_val[0:nx-1, 1:ny, :] +\
                                                bd_vortex_coord_val[1:, 0:ny-1, :]+\
                                                bd_vortex_coord_val[1:, 1:, :])
        bd_vortex_coords = model.create_input('bd_vortex_coords',
                                              val=bd_vortex_coords_val)
        wake_coords = model.create_input('wake_coords', val=wake_coords_val)
        coll_coords = model.create_input('coll_coords', val=coll_pts_val)

        self.add(model, 'inputs')

        self.add(SolveMatrix(nt=nt,bd_vortex_shapes=vor_coord_shapes), name='solve_gamma_b_outputs')

        gamma_b = self.declare_variable('gamma_b',
                                        shape=((nx - 1) * (ny - 1), ))
        frame_vel = self.declare_variable('frame_vel', shape=(3, ))
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
        print('csdl.pnorm(-frame_vel)', (csdl.pnorm(-frame_vel)**2))
        print(
            'delta_p shape-------', delta_p.shape,
            csdl.sum(csdl.reshape(delta_p, new_shape=(n, (nx - 1) * (ny - 1))),
                     axes=(1, )).shape)
        # print('delta_p shape-------', frame_vel_expand.val)
        # print('csdl.pnorm(-frame_vel)**2) shape-------',
        #       (csdl.pnorm(-frame_vel)**2).val)
        # print('delta_p shape-------',
        #       csdl.pnorm(-frame_vel_expand, axis=(1, )).val)
        print('frame_vel___out', frame_vel.val)
        C_l = -csdl.sum(csdl.reshape(delta_p,
                                     new_shape=(n, (nx - 1) * (ny - 1))),
                        axes=(1, )) * np.cos(
                            np.arctan(frame_vel_val[2] / frame_vel_val[0])) / (
                                0.5 * rho * 6) / (csdl.pnorm(-frame_vel)**2)

        # C_l = (csdl.sum(
        #     csdl.reshape(
        #         delta_p, new_shape=(n, (nx - 1) * (ny - 1))))) * np.cos(
        #             10 / 180 * np.pi) / (0.5 * rho *
        #                                  6) / (csdl.pnorm(-frame_vel)**2)
        # print('C_l--------------', C_l.val)
        # delta_p = rho * (
        #     (np.linalg.norm(frame_vel_expand + v_induced, axis=1))**2 -
        #     (np.linalg.norm(frame_vel_expand, axis=1)**2)) / 2

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
        print('C_l--------------', C_l.val)

    def _generate_simple_mesh(self, nx, ny, nt=None):
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


if __name__ == "__main__":
    delta_t = 1
    nt = 5
    nx = 3
    ny = 4

    def generate_simple_mesh(nx, ny, nt=None):
        if nt == None:
            mesh = np.zeros((nx, ny, 3))
            mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny)) + 0.25
            mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
            mesh[:, :, 2] = 0.
        else:
            mesh = np.zeros((nt, nx, ny, 3))
            for i in range(nt):
                mesh[i, :, :, 0] = np.outer(np.arange(nx), np.ones(ny))
                mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
                mesh[i, :, :, 2] = 0.
        return mesh

    model_1 = csdl.Model()

    frame_vel_val = np.array([-1, 0, -1])
    bd_vortex_coords_val = generate_simple_mesh(3, 4)
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

    # coll_val = np.random.random((4, 5, 3))

    model_1.add(ProfileOutputSystemModel())

    sim = Simulator(model_1)
    sim.run()
    sim.visualize_implementation()