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
        self.parameters.declare('num_nodes', types=int)

        self.parameters.declare('eval_pts_option')
        self.parameters.declare('eval_pts_shapes')
        self.parameters.declare('sprs')

        self.parameters.declare('rho', default=0.38)

    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = self.parameters['num_nodes']

        rho = self.parameters['rho']
        sprs = self.parameters['sprs']
        eval_pts_option = self.parameters['eval_pts_option']
        eval_pts_shapes = self.parameters['eval_pts_shapes']

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

        circulations = self.declare_variable('horseshoe_circulation',
                                             shape=(system_size, ))
        circulation_repeat = csdl.expand(circulations, (system_size, 3),
                                         'i->ij')

        # add frame_vel
        frame_vel = self.declare_variable('frame_vel', shape=(num_nodes, 3))
        alpha = csdl.arctan(frame_vel[2] / frame_vel[0])

        if eval_pts_option == 'auto':
            velocities = self.create_output('eval_total_vel',
                                            shape=(system_size, 3))

            start = 0
            for i in range(len(v_total_wake_names)):

                nx = surface_shapes[i][0]
                ny = surface_shapes[i][1]
                delta = (nx - 1) * (ny - 1)
                vel_surface = self.declare_variable(v_total_wake_names[i],
                                                    shape=(delta, 3))
                velocities[start:start + delta, :] = vel_surface
                start = start + delta

            sina = csdl.expand(csdl.sin(alpha), (system_size, 1), 'i->ji')
            cosa = csdl.expand(csdl.cos(alpha), (system_size, 1), 'i->ji')

            panel_forces = rho * circulation_repeat * csdl.cross(
                velocities, bd_vec, axis=1)

            panel_forces_x = panel_forces[:, 0]
            panel_forces_y = panel_forces[:, 1]
            panel_forces_z = panel_forces[:, 2]
            # self.register_output('panel_forces_z', panel_forces_z)
            b = frame_vel[0]**2 + frame_vel[1]**2 + frame_vel[2]**2

            L_panel = -panel_forces_x * sina + panel_forces_z * cosa
            D_panel = panel_forces_x * cosa + panel_forces_z * sina
            start = 0
            for i in range(len(surface_names)):

                mesh = self.declare_variable(surface_names[i],
                                             shape=(1, ) + surface_shapes[i])
                nx = surface_shapes[i][0]
                ny = surface_shapes[i][1]
                #!TODO: need to fix for uniformed mesh - should we take an average?
                chord = csdl.reshape(mesh[:, nx - 1, 0, 0] - mesh[:, 0, 0, 0],
                                     (1, ))
                span = csdl.reshape(mesh[:, 0, ny - 1, 1] - mesh[:, 0, 0, 1],
                                    (1, ))
                L_panel_name = surface_names[i] + '_L_panel'
                D_panel_name = surface_names[i] + '_D_panel'
                L_name = surface_names[i] + '_L'
                D_name = surface_names[i] + '_D'
                CL_name = surface_names[i] + '_C_L'
                CD_name = surface_names[i] + '_C_D_i'

                delta = (nx - 1) * (ny - 1)
                L_panel_surface = L_panel[start:start + delta, :]
                D_panel_surface = D_panel[start:start + delta, :]

                self.register_output(L_panel_name, L_panel_surface)
                self.register_output(D_panel_name, D_panel_surface)
                L = csdl.sum(L_panel_surface, axes=(0, ))
                D = csdl.sum(D_panel_surface, axes=(0, ))
                self.register_output(L_name, csdl.reshape(L, (1, 1)))
                self.register_output(D_name, csdl.reshape(D, (1, 1)))
                c_l = L / (0.5 * rho * span * chord * b)
                c_d = D / (0.5 * rho * span * chord * b)
                self.register_output(CL_name, csdl.reshape(c_l, (1, 1)))
                self.register_output(CD_name, csdl.reshape(c_d, (1, 1)))
                start += delta

        if eval_pts_option == 'user_defined':
            # sina = csdl.expand(csdl.sin(alpha), (system_size, 1), 'i->ji')
            # cosa = csdl.expand(csdl.cos(alpha), (system_size, 1), 'i->ji')

            # panel_forces = rho * circulation_repeat * csdl.cross(
            #     velocities, bd_vec, axis=1)

            # panel_forces_x = panel_forces[:, 0]
            # panel_forces_y = panel_forces[:, 1]
            # panel_forces_z = panel_forces[:, 2]
            # self.register_output('panel_forces_z', panel_forces_z)
            b = frame_vel[0]**2 + frame_vel[1]**2 + frame_vel[2]**2

            # L_panel = -panel_forces_x * sina + panel_forces_z * cosa
            # D_panel = panel_forces_x * cosa + panel_forces_z * sina
            start = 0
            for i in range(len(surface_names)):

                mesh = self.declare_variable(surface_names[i],
                                             shape=(1, ) + surface_shapes[i])
                nx = surface_shapes[i][0]
                ny = surface_shapes[i][1]

                delta = (nx - 1) * (ny - 1)

                nx_eval = eval_pts_shapes[i][0]
                ny_eval = eval_pts_shapes[i][1]
                delta_eval = nx_eval * ny_eval
                vel_surface = self.declare_variable(v_total_wake_names[i],
                                                    shape=(delta_eval, 3))
                # velocities[start:start + delta, :] = vel_surface
                # start = start + delta

                sina = csdl.expand(csdl.sin(alpha), (delta, 1), 'i->ji')
                cosa = csdl.expand(csdl.cos(alpha), (delta, 1), 'i->ji')
                bd_vec_surface = bd_vec[start:start + delta, :]
                circulation_repeat_surface = circulation_repeat[start:start +
                                                                delta, :]
                bd_vec_eval = csdl.sparsematmat(bd_vec_surface, sprs[i])
                sina_eval = csdl.sparsematmat(sina, sprs[i])
                cosa_eval = csdl.sparsematmat(cosa, sprs[i])
                # vel_surface_eval = csdl.sparsematmat(vel_surface, sprs[i])
                circulation_repeat_surface_eval = csdl.sparsematmat(
                    circulation_repeat_surface, sprs[i])

                print('\nbd_vec_eval shape', bd_vec_eval.shape)
                print('vel_surface shape', vel_surface.shape)
                print('circulation_repeat_surface_eval shape',
                      circulation_repeat_surface_eval.shape)

                panel_forces_surface = rho * circulation_repeat_surface_eval * csdl.cross(
                    vel_surface, bd_vec_eval, axis=1)
                panel_forces_x = panel_forces_surface[:, 0]
                panel_forces_y = panel_forces_surface[:, 1]
                panel_forces_z = panel_forces_surface[:, 2]

                chord = csdl.reshape(mesh[:, nx - 1, 0, 0] - mesh[:, 0, 0, 0],
                                     (1, ))
                span = csdl.reshape(mesh[:, 0, ny - 1, 1] - mesh[:, 0, 0, 1],
                                    (1, ))
                L_panel_name = surface_names[i] + '_L_panel'
                D_panel_name = surface_names[i] + '_D_panel'
                L_name = surface_names[i] + '_L'
                D_name = surface_names[i] + '_D'
                CL_name = surface_names[i] + '_C_L'
                CD_name = surface_names[i] + '_C_D_i'

                L_panel_surface = -panel_forces_x * sina_eval + panel_forces_z * cosa_eval
                D_panel_surface = panel_forces_x * cosa_eval + panel_forces_z * sina_eval

                # L_panel_surface = L_panel[start:start + delta, :]
                # D_panel_surface = D_panel[start:start + delta, :]

                self.register_output(L_panel_name, L_panel_surface)
                self.register_output(D_panel_name, D_panel_surface)
                L = csdl.sum(L_panel_surface, axes=(0, ))
                D = csdl.sum(D_panel_surface, axes=(0, ))
                self.register_output(L_name, csdl.reshape(L, (1, 1)))
                self.register_output(D_name, csdl.reshape(D, (1, 1)))
                c_l = L / (0.5 * rho * span * chord * b)
                c_d = D / (0.5 * rho * span * chord * b)
                self.register_output(CL_name, csdl.reshape(c_l, (1, 1)))
                self.register_output(CD_name, csdl.reshape(c_d, (1, 1)))
                start += delta


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
