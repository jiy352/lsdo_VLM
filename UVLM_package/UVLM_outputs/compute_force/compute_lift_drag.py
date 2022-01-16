from csdl_om import Simulator
from csdl import Model
import csdl
from matplotlib.pyplot import clabel
import numpy as np
from numpy.core.fromnumeric import size

from scipy.sparse import csc_matrix


class CL(Model):
    """
    Compute various geometric properties for VLM analysis.
    These are used primarily to help compute postprocessing quantities,
    such as wave CD, viscous CD, etc.
    Some of the quantities, like `normals`, are used to compute the RHS
    of the AIC linear system.
    A \gamma_b = b - M \gamma_w
    parameters
    ----------

    collocation_pts[num_vortex_panel_x*num_vortex_panel_y] : csdl array
        all the bd vertices collocation_pts     
    wake_pts[num_vortex_panel_x*num_vortex_panel_y] : csdl array
        all the wake panel collcation pts 
    wake_circulations[num_wake_panel] : csdl array
        a concatenate vector of the wake circulation strength
    Returns
    -------
    vel_col_w[num_evel_pts_x*num_vortex_panel_x* num_evel_pts_y*num_vortex_panel_y,3]
    csdl array
        the velocities computed using the aic_col_w from biot svart's law
        on bound vertices collcation pts induces by the wakes
    """
    def initialize(self):
        self.parameters.declare('nt', default=5)
        self.parameters.declare('method',
                                values=['fw_euler', 'bk_euler'],
                                default='bk_euler')
        self.parameters.declare('bd_vortex_shapes', types=list)

    def define(self):
        nt = self.parameters['nt']
        bd_vortex_shapes = self.parameters['bd_vortex_shapes']

        system_size = 0
        for i in range(len(bd_vortex_shapes)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            system_size += (nx - 1) * (ny - 1)
        data = [np.ones(system_size)]
        rows = [np.arange(system_size)]
        cols = [np.arange(system_size)]

        ind_1 = 0
        ind_2 = 0

        for i in range(len(bd_vortex_shapes)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            num = (nx - 1) * (ny - 1)

            ind_2 += num

            arange = np.arange(num).reshape((nx - 1), (ny - 1))

            data_ = -np.ones((nx - 2) * (ny - 1))
            rows_ = ind_1 + arange[1:, :].flatten()
            cols_ = ind_1 + arange[:-1, :].flatten()

            data.append(data_)
            rows.append(rows_)
            cols.append(cols_)
            ind_1 += num

        data = np.concatenate(data)
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        mtx_val = csc_matrix((data, (rows, cols)),
                             shape=(system_size, system_size)).toarray()
        mtx = self.create_input('mtx', val=mtx_val)

        gamma_b = self.declare_variable('gamma_b', shape_by_conn=True)
        horseshoe_circulation = csdl.dot(mtx, gamma_b)

        bd_vec = np.zeros((system_size, 3))
        bd_vec[:, 1] = 1
        bd_n_wake = np.concatenate(
            (bd_coords_list[0][0, :, :, :],
             wake_vortex_coords_all[-2].reshape(num_ts, -1, 3)[1:, :, :]))
        rho = 0.38

        sina = np.sin(alpha / 180 * np.pi)
        cosa = np.cos(alpha / 180 * np.pi)
        horseshoe_circulation_repeat = np.einsum('i,j->ij',
                                                 horseshoe_circulation,
                                                 np.ones(3))
        freestream_velocities = frame_vel[-8:, :]
        force_pts_coords = (0.75 * 0.5 * mesh[0:-1, 0:-1, :] +
                            0.75 * 0.5 * mesh[0:-1, 1:, :] +
                            0.25 * 0.5 * mesh[1:, 0:-1, :] +
                            0.25 * 0.5 * mesh[1:, 1:, :])
        aic_force, _ = vel_mtx = InducedVelocity(
            coll_coords_list[0][0, :, :, :],
            bd_n_wake,
            normal_vecs=np.zeros(3),
            compute_aij=False)
        induced_velocities = np.einsum("ijk,j->ik",
                                       aic_force.reshape(8, -1, 3),
                                       circulation_all_current)

        velocities = freestream_velocities + induced_velocities
        panel_forces = rho * circulation_repeat * np.cross(velocities, bd_vec)
        L = np.sum(-panel_forces[:, 0] * sina + panel_forces[:, 2] * cosa)
        b = np.linalg.norm(-frame_vel[i])**2
        C_l = L / (0.5 * rho * span * chord * b)
        self.register_output('C_l', C_l)


# if __name__ == "__main__":

#     def generate_simple_mesh(nx, ny, nt=None):
#         if nt == None:
#             mesh = np.zeros((nx, ny, 3))
#             mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))
#             mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
#             mesh[:, :, 2] = 0.
#         else:
#             mesh = np.zeros((nt, nx, ny, 3))
#             for i in range(nt):
#                 mesh[i, :, :, 0] = np.outer(np.arange(nx), np.ones(ny))
#                 mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
#                 mesh[i, :, :, 2] = 0.
#         return mesh

#     model_1 = Model()

#     frame_vel_val = np.array([1, 0, 1])
#     bd_vortex_coords_val = generate_simple_mesh(3, 4)
#     wake_coords_val = np.array([
#         [2., 0., 0.],
#         [2., 1., 0.],
#         [2., 2., 0.],
#         [2., 3., 0.],
#         [42., 0., 0.],
#         [42., 1., 0.],
#         [42., 2., 0.],
#         [42., 3., 0.],
#     ]).reshape(2, 4, 3)
#     # coll_val = np.random.random((4, 5, 3))

#     frame_vel = model_1.create_input('frame_vel', val=frame_vel_val)
#     bd_vortex_coords = model_1.create_input('bd_vortex_coords',
#                                             val=bd_vortex_coords_val)
#     wake_coords = model_1.create_input('wake_coords', val=wake_coords_val)
#     nx = 3
#     ny = 4
#     coll_pts = 0.25 * (bd_vortex_coords[0:nx-1, 0:ny-1, :] +\
#                                                bd_vortex_coords[0:nx-1, 1:ny, :] +\
#                                                bd_vortex_coords[1:, 0:ny-1, :]+\
#                                                bd_vortex_coords[1:, 1:, :])
#     model_1.register_output('coll_coords', coll_pts)
#     model_1.add(RHS())

#     sim = Simulator(model_1)
#     sim.run()
#     sim.visualize_implementation()
#     # print('aic is', sim['aic'])
#     # print('v_ind is', sim['v_ind'].shape, sim['v_ind'])
