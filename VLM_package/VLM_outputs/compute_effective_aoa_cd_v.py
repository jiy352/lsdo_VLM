from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np


class AOA_CD(Model):
    """
    L,D,cl,cd
    parameters
    ----------

    Returns
    -------
    L,D,cl,cd
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('coeffs_aoa', types=list)
        self.parameters.declare('coeffs_cd', types=list)

    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']

        effective_aoa_names = [x + '_effective_aoa' for x in surface_names]
        cl_span_names = [x + '_cl_span' for x in surface_names]
        cd_v_names = [x + '_cd_v' for x in surface_names]
        cd_span_names = [x + '_cd_i_span' for x in surface_names]
        CD_total_names = [x + '_C_D' for x in surface_names]
        D_total_names = [x + '_D_total' for x in surface_names]
        D_0_names = [x + '_D_0' for x in surface_names]
        num_nodes = surface_shapes[0][0]

        # TODO: fix this rho name
        rho = self.declare_variable('rho',
                                    val=np.ones((num_nodes, 1)) * 0.9652)

        frame_vel = self.declare_variable('frame_vel', shape=(num_nodes, 3))

        # here aoa are all in degrees
        ###! fix for mls
        D_0_list = []

        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            surface_shape = surface_shapes[i]

            # CD_v_name = surface_names[i] + '_C_D_v'
            # CD_name = surface_names[i] + '_C_D'

            effective_aoa_name = effective_aoa_names[i]
            cd_v_name = cd_v_names[i]

            cl_span_name = cl_span_names[i]
            coeff_aoa = self.parameters['coeffs_aoa'][i]
            coeff_cd = self.parameters['coeffs_cd'][i]

            num_span = surface_shape[2] - 1  # ny
            nx = surface_shape[1]

            cl_span = self.declare_variable(cl_span_name,
                                            shape=(num_nodes, num_span))

            effective_aoa = (cl_span - coeff_aoa[0]) / coeff_aoa[1]

            # print('effective_aoa shape', effective_aoa.shape)
            # print('cl_span shape', cl_span.shape)

            # cd_v = csdl.sum(coeff_cd[2] * effective_aoa**2 +
            #                 coeff_cd[1] * effective_aoa + coeff_cd[0],
            #                 axes=(1, )) / num_span
            # print('cd_v shape', cd_v.shape)

            cd_v = coeff_cd[2] * effective_aoa**2 + coeff_cd[
                1] * effective_aoa + coeff_cd[0]

            self.register_output(effective_aoa_name, effective_aoa)
            self.register_output(cd_v_name, cd_v)
            cd_i_span = self.declare_variable(cd_span_names[i],
                                              shape=(num_nodes, num_span))
            cd = csdl.reshape(cd_i_span + cd_v, (num_nodes, num_span, 1))
            s_panels = self.declare_variable(surface_names[i] + '_s_panel',
                                             shape=(num_nodes, nx - 1,
                                                    num_span))

            surface_span = csdl.reshape(csdl.sum(s_panels, axes=(1, )),
                                        (num_nodes, num_span, 1))

            b = frame_vel[:, 0]**2 + frame_vel[:, 1]**2 + frame_vel[:, 2]**2
            rho_b_exp = csdl.expand(rho * b, (num_nodes, num_span, 1),
                                    'ik->ijk')
            # print('shapes\n rho_b_exp', rho_b_exp.shape)
            # print('shapes\n surface_span', surface_span.shape)
            # print('shapes\n cd', cd.shape)
            # print('shapes\n cd',
            #       (csdl.sum(cd * (0.5 * rho_b_exp * surface_span),
            #                 axes=(1, ))).shape)

            # print('shapes\n rho', rho.shape)
            # print('shapes\n b', b.shape)
            # print('shapes\n cd', csdl.sum(surface_span, axes=(1, )).shape)

            C_D_total = csdl.sum(
                cd * (0.5 * rho_b_exp * surface_span), axes=(1, )) / (
                    0.5 * rho * b * csdl.sum(surface_span, axes=(1, )))
            cd_0 = csdl.reshape(cd_v, (num_nodes, num_span, 1))
            D_0 = csdl.sum(cd_0 * (0.5 * rho_b_exp * surface_span), axes=(1, ))
            # C_D = cd_v + self.declare_variable(CD_name, shape=(num_nodes, ))
            self.register_output(CD_total_names[i], C_D_total)
            self.register_output(
                D_total_names[i],
                csdl.sum(cd * (0.5 * rho_b_exp * surface_span), axes=(1, )))
            self.register_output(D_0_names[i], D_0)
            D_0_list.append(D_0)
        D_0_total = self.create_output('D_0_total',
                                       shape=(num_nodes, len(surface_names),
                                              1))

        for i in range(len(surface_names)):
            D_0_total[:, i, :] = csdl.reshape(D_0_list[i],
                                              D_0_total[:, i, :].shape)


if __name__ == "__main__":
    np.random.seed(0)
    nx = 3
    ny = 4
    model_1 = Model()
    surface_names = ['wing']
    surface_shapes = [(3, nx, ny, 3)]

    coeffs_aoa = [(0.535, 0.091)]
    coeffs_cd = [(0.00695, 1.297e-4, 1.466e-4)]

    # coll_val = np.random.random((4, 5, 3))

    wing_cl_span = model_1.create_input('wing_cl_span',
                                        val=np.random.random((3, num_span)))

    model_1.add(
        AOA_CD(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            coeffs_aoa=coeffs_aoa,
            coeffs_cd=coeffs_cd,
        ))

    sim = Simulator(model_1)
    sim.run()
