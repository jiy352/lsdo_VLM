from turtle import shape
from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np

from VLM_package.VLM_preprocessing.compute_bound_vec import BoundVec


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

        self.parameters.declare('rho', default=0.38)

    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        rho = self.parameters['rho']
        effective_aoa_names = [x + '_effective_aoa' for x in surface_names]
        cl_chord_names = [x + '_cl_chord' for x in surface_names]
        cd_v_names = [x + '_cd_v' for x in surface_names]
        # here aoa are all in degrees
        ###! fix for mls

        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            surface_shape = surface_shapes[i]
            effective_aoa_name = effective_aoa_names[i]
            vd_v_name = cd_v_names[i]

            cl_chord_name = cl_chord_names[i]
            coeff_aoa = self.parameters['coeffs_aoa'][i]
            coeff_cd = self.parameters['coeffs_cd'][i]

            num_chord = surface_shape[1] - 1  # ny

            cl_chord = self.declare_variable(cl_chord_name,
                                             shape=(num_chord, ))
            effective_aoa = coeff_aoa[0] * cl_chord + coeff_aoa[1]
            cd_v = csdl.sum(coeff_cd[0] * effective_aoa**2 + coeff_cd[1] *
                            effective_aoa + coeff_cd[2]) / num_chord

            self.register_output(effective_aoa_name, effective_aoa)
            self.register_output(vd_v_name, cd_v)
            C_D = cd_v + self.declare_variable('C_D_i')
            self.register_output('C_D', C_D)


if __name__ == "__main__":

    nx = 3
    ny = 4
    model_1 = Model()
    surface_names = ['wing']
    surface_shapes = [(nx, ny, 3)]

    coeffs_aoa = [np.loadtxt('cl_aoa_coeff.txt')]
    coeffs_cd = [np.loadtxt('cd_aoa_coeff.txt')]
    # coll_val = np.random.random((4, 5, 3))

    wing_cl_chord = model_1.create_input('wing_cl_chord',
                                         val=np.random.random((3 - 1, )))

    model_1.add(
        AOA_CD(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            coeffs_aoa=coeffs_aoa,
            coeffs_cd=coeffs_cd,
        ))

    sim = Simulator(model_1)
    sim.run()
