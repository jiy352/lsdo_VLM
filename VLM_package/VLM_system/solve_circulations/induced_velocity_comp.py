# from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size
# how to declare.options[]? Since I need 'if' statement
# how to set values of the variables outside the class (Indep_var comp?)
# why do we need a create input-how to set its value outside the class
# How to use the same input
# name of the comps, can it be something meaningful
# size differnet than actual python code
# what is the rule for registering outputs
# cannot reshape chords
# prjected vs wetted s_ref?
# line 153 why cannot put 0.5 outside the ()
from VLM_package.VLM_system.solve_circulations.utils.einsum_lijk_lj_lik import EinsumLijkLjLik


class InducedVelocity(Model):
    """
    Compute various geometric properties for VLM analysis.
    These are used primarily to help compute postprocessing quantities,
    such as wave CD, viscous CD, etc.
    Some of the quantities, like `normals`, are used to compute the RHS
    of the AIC linear system.

    parameters
    ----------
    aic[num_evel_pts_x*num_vortex_panel_x* num_evel_pts_y*num_vortex_panel_y,3]
    csdl array
        the AIC matrix computed using biot svart's law
    circulations[num_vortex_panel_x*num_vortex_panel_y] : csdl array
        the circulation strengths of the panels that induces the velocities     

    Returns
    -------
    v_induced[num_evel_pts_x, num_evel_pts_y, 3] : csdl array
        Induced velocities at found along the 3/4 chord.
    """
    def initialize(self):
        self.parameters.declare('aic_names', types=list)
        self.parameters.declare('circulation_names', types=list)

        self.parameters.declare('aic_shapes', types=list)
        self.parameters.declare('circulations_shapes', types=list)

        self.parameters.declare('v_induced_names', types=list)

    def define(self):
        # add_input
        aic_names = self.parameters['aic_names']
        circulation_names = self.parameters['circulation_names']
        aic_shapes = self.parameters['aic_shapes']
        circulations_shapes = self.parameters['circulations_shapes']
        v_induced_names = self.parameters['v_induced_names']

        for i in range(len(aic_names)):

            # input_names
            aic_name = aic_names[i]
            circulations_name = circulation_names[i]

            # output_name
            v_induced_name = v_induced_names[i]

            # input_shapes
            aic_shape = aic_shapes[i]
            circulations_shape = circulations_shapes[i]

            # declare_inputs
            aic = self.declare_variable(aic_name, shape=aic_shape)
            circulations = self.declare_variable(circulations_name,
                                                 shape=circulations_shape)
            aic_reshaped = csdl.reshape(aic,
                                        new_shape=(circulations_shape[0],
                                                   int(aic.shape[1] /
                                                       circulations_shape[1]),
                                                   circulations_shape[1], 3))
            # print('InducedVelocity aic_shape', aic_shape)
            # print('InducedVelocity aic_reshaped', aic_reshaped.shape)
            # print('InducedVelocity circulations_shape', circulations_shape)
            # v_induced = csdl.einsum(
            #     aic_reshaped,
            #     circulations,
            #     subscripts='lijk,lj->lik',
            #     partial_format='dense',
            # )
            self.register_output(aic_name + '_reshaped', aic_reshaped)

            v_induced = csdl.custom(aic_reshaped,
                                    circulations,
                                    op=EinsumLijkLjLik(
                                        in_name_1=aic_name + '_reshaped',
                                        in_name_2=circulations_name,
                                        in_shape=aic_reshaped.shape,
                                        out_name=v_induced_name))

            self.register_output(v_induced_name, v_induced)
            # print('InducedVelocity v_induced shape', v_induced.shape)


if __name__ == "__main__":

    def generate_simple_mesh(nx, ny, n_wake_pts_chord=None):
        if n_wake_pts_chord == None:
            mesh = np.zeros((nx, ny, 3))
            mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))
            mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
            mesh[:, :, 2] = 0.
        else:
            mesh = np.zeros((n_wake_pts_chord, nx, ny, 3))
            for i in range(n_wake_pts_chord):
                mesh[i, :, :, 0] = np.outer(np.arange(nx), np.ones(ny))
                mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
                mesh[i, :, :, 2] = 0.
        return mesh

    aic_names = ['aic']
    circulation_names = ['circ']
    aic_shapes = [(36, 3)]
    circulations_shapes = [(6)]
    v_induced_names = ['v_ind']

    model_1 = Model()
    aic_val = np.random.random((36, 3))
    circulations_val = np.random.random((6))

    vor = model_1.create_input('aic', val=aic_val)
    col = model_1.create_input('circ', val=circulations_val)
    model_1.add(InducedVelocity(aic_names=aic_names,
                                circulation_names=circulation_names,
                                aic_shapes=aic_shapes,
                                circulations_shapes=circulations_shapes,
                                v_induced_names=v_induced_names),
                name='Vind_comp')
    sim = Simulator(model_1)
    sim.run()
    print('aic is', sim['aic'])
    print('v_ind is', sim['v_ind'].shape, sim['v_ind'])
