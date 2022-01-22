from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np


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

    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        rho = self.parameters['rho']
        v_total_wake_names = [x + '_eval_total_vel' for x in surface_names]
        system_size = 0
        for i in range(len(surface_names)):
            nx = surface_shapes[i][0]
            ny = surface_shapes[i][1]
            system_size += (nx - 1) * (ny - 1)

        # bd_vec_model = Model()
        # !TODO!: fix this for variable mesh resolutiopn, and for inclined meshes
        bd_vec_val = np.zeros((system_size, 3))
        bd_vec_val[:, 1] = -1
        bd_vec = self.declare_variable('bd_vec', bd_vec_val)
        chord = nx - 1
        span = ny - 1
        # self.add(bd_vec_model, name='compute_bd_vec')

        # add circulations and force point velocities

        circulations = self.declare_variable('horseshoe_circulation',
                                             shape=(system_size, ))
        circulation_repeat = csdl.expand(circulations, (system_size, 3),
                                         'i->ij')
        # !TODO: fix this for mls
        # print('name_in_LD', v_induced_wake_names[0])
        velocities = self.declare_variable(v_total_wake_names[0],
                                           shape=(system_size, 3))
        # add frame_vel
        frame_vel = self.declare_variable('frame_vel', shape=(3, ))

        alpha = csdl.arctan(frame_vel[2] / frame_vel[0])
        sina = csdl.expand(csdl.sin(alpha), (system_size, 1), 'i->ji')
        cosa = csdl.expand(csdl.cos(alpha), (system_size, 1), 'i->ji')

        panel_forces = rho * circulation_repeat * csdl.cross(
            velocities, bd_vec, axis=1)

        panel_forces_x = panel_forces[:, 0]
        panel_forces_z = panel_forces[:, 2]

        L = csdl.sum(-panel_forces_x * sina + panel_forces_z * cosa,
                     axes=(0, ))
        D = csdl.sum(-panel_forces_x * cosa + panel_forces_z * sina,
                     axes=(0, ))
        b = frame_vel[0]**2 + frame_vel[1]**2 + frame_vel[2]**2

        # print(panel_forces[:, 0].shape)
        # print(frame_vel.shape)
        # print(
        #     'b',
        #     b.shape,
        # )
        # print(L.shape)

        c_l = L / (0.5 * rho * span * chord * b)
        c_d = D / (0.5 * rho * span * chord * b)
        self.register_output('L', csdl.reshape(L, (1, 1)))
        self.register_output('D', csdl.reshape(D, (1, 1)))
        self.register_output('C_L', csdl.reshape(c_l, (1, 1)))
        self.register_output('C_D', csdl.reshape(c_d, (1, 1)))


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
