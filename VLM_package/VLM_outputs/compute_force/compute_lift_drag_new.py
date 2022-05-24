from turtle import shape
from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np

from VLM_package.VLM_preprocessing.compute_bound_vec import BoundVec


class LiftDrag(Model):
    """
    L,D,cl,cd
    parameters
    ----------

    bd_vec : csdl array
        tangential vec    
    velocities: csdl array
        force_pts vel 
    gamma_b[num_bd_panel] : csdl array
        a concatenate vector of the bd circulation strength
    frame_vel[3,] : csdl array
        frame velocities
    Returns
    -------
    L,D,cl,cd
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)

        self.parameters.declare('rho', default=0.38)

        self.parameters.declare('eval_pts_shapes', types=list)
        self.parameters.declare('eval_pts_ind', types=list)
        self.parameters.declare('sprs', types=list)

    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        eval_pts_shapes = self.parameters['eval_pts_shapes']
        eval_pts_ind = self.parameters['eval_pts_ind']
        sprs = self.parameters['sprs']

        rho = self.parameters['rho']
        v_total_wake_names = [x + '_eval_total_vel' for x in surface_names]
        system_size = 0

        for i in range(len(surface_names)):
            nx = surface_shapes[i][0]
            ny = surface_shapes[i][1]
            system_size += (nx - 1) * (ny - 1)

        submodel = BoundVec(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
        )
        self.add(submodel, name='BoundVec')

        bd_vec = self.declare_variable('bd_vec', shape=((system_size, 3)))
        mesh = self.declare_variable(surface_names[0],
                                     shape=(1, ) + surface_shapes[0])
        chord = csdl.reshape(mesh[:, nx - 1, 0, 0] - mesh[:, 0, 0, 0], (1, ))
        span = csdl.reshape(mesh[:, 0, ny - 1, 1] - mesh[:, 0, 0, 1], (1, ))

        # add circulations and force point velocities

        circulations = self.declare_variable('horseshoe_circulation',
                                             shape=(system_size, ))
        circulation_repeat = csdl.expand(circulations, (system_size, 3),
                                         'i->ij')
        # !TODO: fix this for mls
        # print('name_in_LD', v_induced_wake_names[0])
        total_size = 0
        for i in range(len(v_total_wake_names)):
            nx = eval_pts_shapes[i][0]
            ny = eval_pts_shapes[i][1]
            total_size += nx * ny

        start = 0

        for i in range(len(v_total_wake_names)):
            velocities = self.create_output('eval_total_vel',
                                            shape=(total_size, 3))
            nx = eval_pts_shapes[i][0]
            ny = eval_pts_shapes[i][1]
            delta = nx * ny

            vel_surface = self.declare_variable(v_total_wake_names[i],
                                                shape=(delta, 3))
            velocities[start:start + delta, :] = vel_surface
            start = start + delta

        # add frame_vel
        frame_vel = self.declare_variable('frame_vel', shape=(3, ))

        alpha = csdl.arctan(frame_vel[2] / frame_vel[0])
        sina = csdl.expand(csdl.sin(alpha), (system_size, 1), 'i->ji')
        cosa = csdl.expand(csdl.cos(alpha), (system_size, 1), 'i->ji')

        print('velocities_______________000000000000', velocities.shape)
        print('system_size', system_size)

        print('bd_vec_______________000000000000', bd_vec.shape)
        print('velocities_______________000000000000',
              circulation_repeat.shape)

        for i in range(len(surface_names)):
            circulation_repeat_eval = csdl.sparsematmat(
                circulation_repeat, sprs[i])
            bd_vec_eval = csdl.sparsematmat(bd_vec, sprs[i])
            sina_eval = csdl.sparsematmat(sina, sprs[i])
            cosa_eval = csdl.sparsematmat(cosa, sprs[i])
            s_panel = self.declare_variable(surface_names[i] + '_s_panel',
                                            shape=(1, surface_shapes[i][0] - 1,
                                                   surface_shapes[i][1] - 1))
            print('s_panel', s_panel.shape)
            print('sina', sina.shape)
            print(' sprs[i]', sprs[i].shape)
            print(
                's_panel',
                csdl.reshape(s_panel, ((surface_shapes[i][0] - 1) *
                                       (surface_shapes[i][1] - 1), )).shape)
            area_eval = csdl.sparsematmat(
                csdl.reshape(s_panel, ((surface_shapes[i][0] - 1) *
                                       (surface_shapes[i][1] - 1), 1)),
                sprs[i])

            print('circulation_repeat_eval', circulation_repeat_eval.shape)
            print('area_eval', area_eval.shape)

            panel_forces = rho * circulation_repeat_eval * csdl.cross(
                velocities, bd_vec_eval, axis=1) / csdl.expand(
                    csdl.reshape(area_eval, (area_eval.shape[0])),
                    (area_eval.shape[0], 3), 'i->ij')
            # print('panel_forces', panel_forces.shape)

            panel_forces_x = panel_forces[:, 0]
            panel_forces_y = panel_forces[:, 1]
            panel_forces_z = panel_forces[:, 2]
            # self.register_output('bd_vec', bd_vec)
            self.register_output(surface_names[i] + '_panel_forces',
                                 panel_forces)

            L = csdl.sum(-panel_forces_x * sina_eval +
                         panel_forces_z * cosa_eval,
                         axes=(0, ))
            # !TODO:! need to check the sign here
            print('shapes')
            print('panel_forces', panel_forces.shape, panel_forces_x.shape)

            D = csdl.sum(panel_forces_x * cosa_eval +
                         panel_forces_z * sina_eval,
                         axes=(0, ))
            b = frame_vel[0]**2 + frame_vel[1]**2 + frame_vel[2]**2

            c_l = L / (0.5 * rho * span * chord * b)
            c_d = D / (0.5 * rho * span * chord * b)
            self.register_output('L', csdl.reshape(L, (1, 1)))
            self.register_output('D', csdl.reshape(D, (1, 1)))
            self.register_output('C_L', csdl.reshape(c_l, (1, 1)))
            self.register_output('C_D_i', csdl.reshape(c_d, (1, 1)))


if __name__ == "__main__":

    nx = 3
    ny = 4
    model_1 = Model()
    surface_names = ['wing']
    surface_shapes = [(nx, ny, 3)]

    frame_vel_val = np.array([-1, 0, -1])
    f_val = np.einsum(
        'i,j->ij',
        np.ones(6),
        np.array([-1, 0, -1]) + 1e-3,
    )

    # coll_val = np.random.random((4, 5, 3))

    frame_vel = model_1.create_input('frame_vel', val=frame_vel_val)
    gamma_b = model_1.create_input('gamma_b',
                                   val=np.random.random(((nx - 1) * (ny - 1))))
    force_pt_vel = model_1.create_input('force_pt_vel', val=f_val)

    model_1.add(
        LiftDrag(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
        ))

    frame_vel = model_1.declare_variable('L', shape=(1, ))
    frame_vel = model_1.declare_variable('D', shape=(1, ))

    sim = Simulator(model_1)
    sim.run()
