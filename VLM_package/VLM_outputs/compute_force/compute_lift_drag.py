# from turtle import shape
# from csdl_om import Simulator
from csdl import Model
import csdl
from matplotlib.pyplot import axis
import numpy as np

from VLM_package.VLM_outputs.compute_effective_aoa_cd_v import AOA_CD


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

        self.parameters.declare('eval_pts_option')
        self.parameters.declare('eval_pts_shapes')
        self.parameters.declare('sprs')

        # self.parameters.declare('rho', default=0.9652)
        self.parameters.declare('eval_pts_names', types=None)

        self.parameters.declare('coeffs_aoa', default=None)
        self.parameters.declare('coeffs_cd', default=None)
        self.parameters.declare('cl0', types=list)

    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        cl0 = self.parameters['cl0']
        num_nodes = surface_shapes[0][0]
        frame_vel = self.declare_variable('frame_vel', shape=(num_nodes, 3))

        cl_span_names = [x + '_cl_span' for x in surface_names]

        system_size = 0
        for i in range(len(surface_names)):
            nx = surface_shapes[i][1]
            ny = surface_shapes[i][2]
            system_size += (nx - 1) * (ny - 1)

        # rho = self.declare_variable('rho', shape=(num_nodes, 1))
        rho = self.declare_variable('density', shape=(num_nodes, 1))
        # self.print_var(rho)
        rho_expand = csdl.expand(csdl.reshape(rho, (num_nodes, )),
                                 (num_nodes, system_size, 3), 'k->kij')
        # self.print_var(rho_expand)
        alpha = self.declare_variable('alpha', shape=(num_nodes, 1))
        beta = self.declare_variable('beta', shape=(num_nodes, 1))

        sprs = self.parameters['sprs']
        eval_pts_option = self.parameters['eval_pts_option']
        eval_pts_shapes = self.parameters['eval_pts_shapes']

        coeffs_aoa = self.parameters['coeffs_aoa']
        coeffs_cd = self.parameters['coeffs_cd']

        if eval_pts_option=='auto':
            eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
        else:
            eval_pts_names=self.parameters['eval_pts_names']
        v_total_wake_names = [x + '_eval_total_vel' for x in eval_pts_names]

        bd_vec = self.declare_variable('bd_vec',
                                       shape=((num_nodes, system_size, 3)))

        circulations = self.declare_variable('horseshoe_circulation',
                                             shape=(num_nodes, system_size))
        circulation_repeat = csdl.expand(circulations,
                                         (num_nodes, system_size, 3),
                                         'ki->kij')

        # print('beta shape', beta.shape)
        # print('sinbeta shape', sinbeta.shape)

        eval_pts_names = self.parameters['eval_pts_names']

        if eval_pts_option == 'auto':
            velocities = self.create_output('eval_total_vel',
                                            shape=(num_nodes, system_size, 3))
            s_panels_all = self.create_output('s_panels_all',
                                              shape=(num_nodes, system_size))
            eval_pts_all = self.create_output('eval_pts_all',
                                              shape=(num_nodes, system_size,
                                                     3))
            start = 0
            for i in range(len(v_total_wake_names)):

                nx = surface_shapes[i][1]
                ny = surface_shapes[i][2]
                delta = (nx - 1) * (ny - 1)
                vel_surface = self.declare_variable(v_total_wake_names[i],
                                                    shape=(num_nodes, delta,
                                                           3))
                s_panels = self.declare_variable(surface_names[i] + '_s_panel',
                                                 shape=(num_nodes, nx - 1,
                                                        ny - 1))

                spans = self.declare_variable(
                    surface_names[i] + '_span_length',
                    shape=(num_nodes, nx - 1, ny - 1))
                chords = self.declare_variable(
                    surface_names[i] + '_chord_length',
                    shape=(num_nodes, nx - 1, ny - 1))
                eval_pts = self.declare_variable(eval_pts_names[i],
                                                 shape=(num_nodes, nx - 1,
                                                        ny - 1, 3))
                # print('compute lift drag vel_surface shape', vel_surface.shape)
                # print('compute lift drag velocities shape', velocities.shape)
                velocities[:, start:start + delta, :] = vel_surface
                s_panels_all[:, start:start + delta] = csdl.reshape(
                    s_panels, (num_nodes, delta))
                eval_pts_all[:, start:start + delta, :] = csdl.reshape(
                    eval_pts, (num_nodes, delta, 3)) 
                # spans_all[:, start:start + delta] = csdl.reshape(
                #     spans, (num_nodes, delta))
                # chords_all[:, start:start + delta] = csdl.reshape(
                #     chords, (num_nodes, delta))
                start = start + delta

            # print('-----------------alpha', alpha.name, csdl.cos(alpha).name)
            # print('-----------------beta', beta.name, csdl.cos(beta).name)

            sina = csdl.expand(csdl.sin(alpha), (num_nodes, system_size, 1),
                               'ki->kji')
            cosa = csdl.expand(csdl.cos(alpha), (num_nodes, system_size, 1),
                               'ki->kji')
            sinb = csdl.expand(csdl.sin(beta), (num_nodes, system_size, 1),
                               'ki->kji')
            cosb = csdl.expand(csdl.cos(beta), (num_nodes, system_size, 1),
                               'ki->kji')
            # print('-----------------cosa', cosa.name)
            # print('-----------------sinb', sinb.name)
            # print('-----------------cosb', cosb.name)

            panel_forces = rho_expand * circulation_repeat * csdl.cross(
                velocities, bd_vec, axis=2)

            self.register_output('panel_forces', panel_forces)

            panel_forces_x = panel_forces[:, :, 0]
            panel_forces_y = panel_forces[:, :, 1]
            panel_forces_z = panel_forces[:, :, 2]
            # print('compute lift drag panel_forces', panel_forces.shape)
            b = frame_vel[:, 0]**2 + frame_vel[:, 1]**2 + frame_vel[:, 2]**2

            L_panel = -panel_forces_x * sina + panel_forces_z * cosa
            D_panel = panel_forces_x * cosa * cosb + panel_forces_z * sina * cosb - panel_forces_y * sinb
            traction_panel = panel_forces / csdl.expand(
                s_panels_all, panel_forces.shape, 'ij->ijk')

            s_panels_sum = csdl.reshape(csdl.sum(s_panels_all, axes=(1, )),
                                        (num_nodes, 1))

            start = 0
            for i in range(len(surface_names)):

                # mesh = self.declare_variable(surface_names[i],
                #                              shape=surface_shapes[i])
                nx = surface_shapes[i][1]
                ny = surface_shapes[i][2]

                # s_panels = self.declare_variable(surface_names[i] + '_s_panel',
                #                                  shape=(num_nodes, nx - 1,
                #                                         ny - 1))

                # nx = surface_shapes[i][1]
                # ny = surface_shapes[i][2]
                #!TODO: need to fix for uniformed mesh - should we take an average?
                # chord = csdl.reshape(mesh[:, nx - 1, 0, 0] - mesh[:, 0, 0, 0],
                #                      (num_nodes, 1))
                # span = csdl.reshape(mesh[:, 0, ny - 1, 1] - mesh[:, 0, 0, 1],
                #                     (num_nodes, 1))
                L_panel_name = surface_names[i] + '_L_panel'
                D_panel_name = surface_names[i] + '_D_panel'
                traction_surfaces_name = surface_names[i] + '_traction_surfaces'

                L_name = surface_names[i] + '_L'
                D_name = surface_names[i] + '_D'
                CL_name = surface_names[i] + '_C_L'
                CD_name = surface_names[i] + '_C_D_i'

                delta = (nx - 1) * (ny - 1)
                L_panel_surface = L_panel[:, start:start + delta, :]
                D_panel_surface = D_panel[:, start:start + delta, :]
                # cl_panel_surface = cl_panel[:, start:start + delta, :]
                # cdi_panel_surface = cd_i_panel[:, start:start + delta, :]
                traction_surfaces = traction_panel[:, start:start + delta, :]

                self.register_output(L_panel_name, L_panel_surface)
                self.register_output(D_panel_name, D_panel_surface)
                self.register_output(traction_surfaces_name, traction_surfaces)

                L = csdl.sum(L_panel_surface, axes=(1, ))
                D = csdl.sum(D_panel_surface, axes=(1, ))
                self.register_output(L_name, csdl.reshape(L, (num_nodes, 1)))
                self.register_output(D_name, csdl.reshape(D, (num_nodes, 1)))

                c_l = L / (0.5 * rho * s_panels_sum * b)
                self.register_output(CL_name,
                                     csdl.reshape(c_l, (num_nodes, 1)))

                c_d = D / (0.5 * rho * s_panels_sum * b)

                self.register_output(CD_name,
                                     csdl.reshape(c_d, (num_nodes, 1)))

                start += delta

            if self.parameters['coeffs_aoa'] != None:
                # print('coeffs_aoa is ', self.parameters['coeffs_aoa'])
                cl_span_names = [x + '_cl_span' for x in surface_names]
                cd_span_names = [x + '_cd_i_span' for x in surface_names]
                # nx = surface_shapes[i][1]
                # ny = surface_shapes[i][2]
                start = 0
                for i in range(len(surface_names)):
                    nx = surface_shapes[i][1]
                    ny = surface_shapes[i][2]
                    delta = (nx - 1) * (ny - 1)

                    s_panels = self.declare_variable(
                        surface_names[i] + '_s_panel',
                        shape=(num_nodes, nx - 1, ny - 1))
                    surface_span = csdl.reshape(csdl.sum(s_panels, axes=(1, )),
                                                (num_nodes, ny - 1, 1))
                    rho_b_exp = csdl.expand(rho * b, (num_nodes, ny - 1, 1),
                                            'ik->ijk')

                    cl_span = csdl.reshape(
                        csdl.sum(csdl.reshape(
                            L_panel[:, start:start + delta, :],
                            (num_nodes, nx - 1, ny - 1)),
                                 axes=(1, )),
                        (num_nodes, ny - 1, 1)) / (0.5 * rho_b_exp *
                                                   surface_span)
                    # print()
                    cd_span = csdl.reshape(
                        csdl.sum(csdl.reshape(
                            D_panel[:, start:start + delta, :],
                            (num_nodes, nx - 1, ny - 1)),
                                 axes=(1, )),
                        (num_nodes, ny - 1, 1)) / (0.5 * rho_b_exp *
                                                   surface_span)
                    self.register_output(cl_span_names[i], cl_span)
                    self.register_output(cd_span_names[i], cd_span)
                    start += delta

                sub = AOA_CD(
                    surface_names=surface_names,
                    surface_shapes=surface_shapes,
                    coeffs_aoa=coeffs_aoa,
                    coeffs_cd=coeffs_cd,
                )
                self.add(sub, name='AOA_CD')

                #     cd_v_names = [x + '_cd_v' for x in surface_names]

                #     for i in range(len(surface_names)):
                D_total_name = surface_names[i] + '_D_total'

                #         cd_v = self.declare_variable(cd_v_names[i],
                #                                      shape=(num_nodes, 1))
                #         c_d_total = cd_v + c_d
                CD_total_names = [x + '_C_D' for x in surface_names]
                # for i in range(len(surface_names)):

                #     c_d_total = self.declare_variable(CD_total_names[i],
                #                                       shape=(num_nodes, 1))

                #     D_total = c_d_total * (0.5 * rho * s_panels_sum * b)
                # self.register_output(D_total_name, D_total)

            ##########################################################
            # temp fix total_forces = csdl.sum(panel_forces, axes=(1, ))
            ##########################################################
            # print('D shape', D.shape)
            # print('L shape', L.shape)
            # total_forces = self.create_output('F', shape=(num_nodes, 3))
            # total_forces[:, 0] = -D
            # total_forces[:, 2] = L

            # print('shapes total force', total_forces.shape)
            # print('shapes panel_forces', panel_forces.shape)
            # print('shapes eval_pts_all', eval_pts_all.shape)
            # print('shapes eval_pts_all',
            #       csdl.cross(panel_forces, eval_pts_all, axis=(2, )))

            #TODO: discuss about the drag computation
            D_0_total = self.declare_variable('D_0_total',
                                              val=np.zeros((num_nodes,
                                                     len(surface_names), 1)))
            D_0 = csdl.sum(D_0_total, axes=(1, ))

            L_0 = self.create_output('L_0',shape=(num_nodes,len(v_total_wake_names))) 
            # L_0 is size num_nodes by num_surfaces

            for i in range(len(v_total_wake_names)):

                nx = surface_shapes[i][1]
                ny = surface_shapes[i][2]
                delta = (nx - 1) * (ny - 1)
                C_L_0 = cl0[i]
                vel_surface = self.declare_variable(v_total_wake_names[i],
                                                    shape=(num_nodes, delta,
                                                           3))
                s_panels = self.declare_variable(surface_names[i] + '_s_panel',
                                                 shape=(num_nodes, nx - 1,
                                                        ny - 1))
                s_panels_sum_surface = csdl.reshape(csdl.sum(s_panels,axes=(1,2,)),(num_nodes,1))

                L_0[:,i] = 0.5 *rho*b*s_panels_sum_surface*C_L_0
            L_0_total = csdl.reshape((csdl.sum(L_0,axes=(1,))),(num_nodes,1))
                

            self.register_output('viscous_drag',D_0)
            # self.print_var(D_0)
            # self.print_var(L_0)

            total_forces_temp = csdl.sum(panel_forces, axes=(1, ))
            F = self.create_output('F', shape=(num_nodes, 3))
            # print('D_0.shape', D_0.shape)
            # print('L_0_total',L_0_total.shape)
            # print('alpha',alpha.shape)
            # print('s_panels_sum_surface',s_panels_sum_surface.shape)
            # print('C_L_0',C_L_0)
            # self.print_var(D_0 * csdl.cos(alpha))
            drag_coeff = 9 * (0.092903)
            fuselage_drag = 0.5*rho*b*drag_coeff
            self.register_output('fuselage_drag',fuselage_drag)
            # self.print_var(fuselage_drag)
            F[:, 0] = -(total_forces_temp[:, 0] + D_0 * csdl.cos(alpha) - L_0_total * csdl.sin(alpha) + fuselage_drag* csdl.cos(alpha))
            F[:, 1] = total_forces_temp[:, 1] * 0
            F[:, 2] = -(total_forces_temp[:, 2] + D_0 * csdl.sin(alpha) + L_0_total * csdl.cos(alpha))

            ''' Testing forces computation '''
            # tempD0Total = D_0_total * np.ones((num_nodes,len(surface_names), 1))
            # self.register_output('TempD0Total', tempD0Total)
            # self.print_var(tempD0Total)

            tempF = F * np.ones((num_nodes, 3))
            self.register_output('Forces_IN_VLM', tempF)
            self.print_var(tempF)

            # tempTotalForces = total_forces_temp * np.ones((num_nodes, 3))
            # self.register_output('TotalForces_IN_VLM', tempTotalForces)
            # self.print_var(tempTotalForces)

            # tempAlpha = alpha * np.ones((num_nodes, 1))
            # self.register_output('TempAlpha_IN_VLM', tempAlpha)
            # self.print_var(tempAlpha)

            # tempL0Total = L_0_total * np.ones((num_nodes, 1))
            # self.register_output('TempL0total_IN_VLM', tempL0Total)
            # self.print_var(tempL0Total)

            # tempRho = rho*np.ones((num_nodes, 1))
            # self.register_output('TempRho_IN_VLM', tempRho)
            # self.print_var(tempRho)

            # tempB = b*np.ones((num_nodes, 1))
            # self.register_output('Tempb_IN_VLM', tempB)
            # self.print_var(tempB)







            evaluation_pt = self.declare_variable('evaluation_pt',
                                                  val=np.zeros(3, )) 
            evaluation_pt_exp = csdl.expand(
                evaluation_pt,
                (eval_pts_all.shape),
                'i->jki',
            )
            r_M = eval_pts_all - evaluation_pt_exp
            total_moment = csdl.sum(csdl.cross(r_M, panel_forces, axis=2),
                                    axes=(1, ))
            M = self.create_output('M', shape=total_moment.shape)

            M[:, 0] = total_moment[:, 0] * 0
            M[:, 1] = -total_moment[:, 1]
            M[:, 2] = total_moment[:, 2] * 0
            '''hardcode for testing'''

            D_total = -F[:, 0]*csdl.cos(alpha) - F[:, 2]*csdl.sin(alpha)
            L_total =  F[:, 0]*csdl.sin(alpha) - F[:, 2]*csdl.cos(alpha)
            C_D_total = D_total/(0.5 *rho*b*s_panels_sum_surface)
            C_L_total = L_total/(0.5 *rho*b*s_panels_sum_surface)           
            self.register_output('total_drag', D_total)
            self.register_output('total_lift', L_total)
            L_over_D = L_total / D_total
            self.register_output('L_over_D', L_over_D)
            self.print_var(L_over_D)
            self.register_output('total_CD', C_D_total)
            self.register_output('total_CL', C_L_total)
            # else:
            #     for i in range(len(surface_names)):
            #         D_total_name = surface_names[i] + '_D_total'
            #     self.register_output(D_total_name, D + 0)

        # !TODO: need to fix eval_pts for main branch
        if eval_pts_option == 'user_defined':
            v_total_eval_names = [x + '_eval_total_vel' for x in eval_pts_names]
            print('v_total_eval_names',v_total_eval_names)
            eval_vel_shapes = [(num_nodes, x[1] * x[2], 3) for x in eval_pts_shapes]
            # sina = csdl.expand(csdl.sin(alpha), (system_size, 1), 'i->ji')
            # cosa = csdl.expand(csdl.cos(alpha), (system_size, 1), 'i->ji')

            # panel_forces = rho * circulation_repeat * csdl.cross(
            #     velocities, bd_vec, axis=1)

            # panel_forces_x = panel_forces[:, 0]
            # panel_forces_y = panel_forces[:, 1]
            # panel_forces_z = panel_forces[:, 2]
            # self.register_output('panel_forces_z', panel_forces_z)

            # L_panel = -panel_forces_x * sina + panel_forces_z * cosa
            # D_panel = panel_forces_x * cosa + panel_forces_z * sina

            start = 0
            for i in range(len(eval_vel_shapes)):
                v_total_eval = self.declare_variable(v_total_eval_names[i],shape=eval_vel_shapes[i])
                rho_exp_1 = csdl.expand(csdl.reshape(rho,new_shape=(num_nodes,)), (eval_vel_shapes[i][0], eval_vel_shapes[i][1]),'i->ij')
                print('rho_exp_1',rho_exp_1.shape)
                print('csdl.sum(v_total_eval**2, axes=(2,))',csdl.sum(v_total_eval**2, axes=(2,)).shape)

                dynamic_presure = 0.5*rho_exp_1*(csdl.sum(v_total_eval**2, axes=(2,)))/1000
                freestream_pressure = 0.5*rho*(frame_vel[:, 0]**2 + frame_vel[:, 1]**2 + frame_vel[:, 2]**2)/1000
                air_pressure = self.declare_variable('pressure', shape=(num_nodes,1))
                air_pressure_flatten = csdl.reshape(air_pressure, (num_nodes,))
                freestream_pressure_flatten = csdl.reshape(freestream_pressure, (num_nodes,))
                static_pressure = csdl.expand(air_pressure_flatten, (num_nodes, eval_vel_shapes[i][1]),'i->ij') - dynamic_presure #  kpa
                static_pressure_diff = csdl.expand(freestream_pressure_flatten, (num_nodes, eval_vel_shapes[i][1]),'i->ij') - dynamic_presure #  kpa
                pressure_coeff = static_pressure_diff/csdl.expand(freestream_pressure_flatten, (num_nodes, eval_vel_shapes[i][1]),'i->ij')
                print('dynamic_presure',dynamic_presure.shape,dynamic_presure.name)
                self.register_output(eval_pts_names[i]+'_dynamic_pressure',dynamic_presure)
                self.register_output(eval_pts_names[i]+'_static_pressure',static_pressure)
                self.register_output(eval_pts_names[i]+'_static_pressure_diff',static_pressure_diff)
                self.register_output(eval_pts_names[i]+'_pressure_coeff', pressure_coeff)




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
