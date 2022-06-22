from csdl_om import Simulator
from csdl import Model
import csdl
from matplotlib.pyplot import clabel
import numpy as np
from numpy.core.fromnumeric import size

from scipy.sparse import csc_matrix

from VLM_package.VLM_outputs.compute_force.horseshoe_circulations import HorseshoeCirculations
from VLM_package.VLM_outputs.compute_force.eval_pts_velocities_mls import EvalPtsVel
from VLM_package.VLM_outputs.compute_force.compute_lift_drag import LiftDrag


class Outputs(Model):
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
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)

        self.parameters.declare('eval_pts_names', types=list)

        self.parameters.declare('eval_pts_location')
        self.parameters.declare('eval_pts_option')
        self.parameters.declare('eval_pts_shapes', types=list)
        self.parameters.declare('sprs')

        self.parameters.declare('n_wake_pts_chord', default=2)
        self.parameters.declare('delta_t', default=100)

        self.parameters.declare('coeffs_aoa', default=None)
        self.parameters.declare('coeffs_cd', default=None)
        self.parameters.declare('AcStates', default=None)

    def define(self):
        n_wake_pts_chord = self.parameters['n_wake_pts_chord']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']

        eval_pts_names = self.parameters['eval_pts_names']
        eval_pts_shapes = self.parameters['eval_pts_shapes']
        eval_pts_option = self.parameters['eval_pts_option']
        eval_pts_location = self.parameters['eval_pts_location']
        sprs = self.parameters['sprs']

        delta_t = self.parameters['delta_t']
        coeffs_aoa = self.parameters['coeffs_aoa']
        coeffs_cd = self.parameters['coeffs_cd']

        submodel = HorseshoeCirculations(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
        )
        self.add(submodel, name='compute_horseshoe_circulation')

        submodel = EvalPtsVel(
            eval_pts_names=eval_pts_names,
            eval_pts_shapes=eval_pts_shapes,
            eval_pts_option=eval_pts_option,
            eval_pts_location=eval_pts_location,
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            n_wake_pts_chord=n_wake_pts_chord,
            delta_t=delta_t,
        )
        self.add(submodel, name='EvalPtsVel')

        submodel = LiftDrag(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            eval_pts_option=eval_pts_option,
            eval_pts_shapes=eval_pts_shapes,
            eval_pts_names=eval_pts_names,
            sprs=sprs,
            coeffs_aoa=coeffs_aoa,
            coeffs_cd=coeffs_cd,
            AcStates=self.parameters['AcStates'],
        )
        self.add(submodel, name='LiftDrag')


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

    nx = 3
    ny = 4
    n_wake_pts_chord = 5
    delta_t = 1
    # surface_names = ['wing']
    # surface_shapes = [(nx, ny, 3)]
    # eval_pts_names = ['eval_pts_coords']
    # eval_pts_shapes = [(nx - 1, ny - 1, 3)]

    surface_names = ['wing1', 'wing2']
    surface_shapes = [(3, 4, 3), (4, 5, 3)]
    eval_pts_names = ['wing1_force_pts', 'wing2_force_pts']
    eval_pts_shapes = [(2, 3, 3), (3, 4, 3)]

    model_1 = Model()

    model_1.add(
        Outputs(surface_names=surface_names,
                surface_shapes=surface_shapes,
                n_wake_pts_chord=n_wake_pts_chord,
                eval_pts_names=eval_pts_names,
                eval_pts_shapes=eval_pts_shapes,
                delta_t=delta_t))

    sim = Simulator(model_1)
    sim.run()
    sim.visualize_implementation()
#     # print('aic is', sim['aic'])
#     # print('v_ind is', sim['v_ind'].shape, sim['v_ind'])
