from csdl_om import Simulator
from csdl import Model
import csdl
from matplotlib.pyplot import clabel
import numpy as np
from numpy.core.fromnumeric import size

from scipy.sparse import csc_matrix

from UVLM_package.UVLM_outputs.compute_force.horseshoe_circulations import HorseshoeCirculations
from UVLM_package.UVLM_outputs.compute_force.eval_pts_velocities import EvalPtsVel
from UVLM_package.UVLM_outputs.compute_force.compute_lift_drag import LiftDrag


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
        self.parameters.declare('nt')

        self.parameters.declare('eval_pts_names', types=list)
        self.parameters.declare('eval_pts_shapes', types=list)
        self.parameters.declare('eval_pts_location', default=0.25)

        # stands for quarter-chord
        self.parameters.declare('nt')
        self.parameters.declare('delta_t')

    def define(self):
        nt = self.parameters['nt']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']

        eval_pts_names = self.parameters['eval_pts_names']
        eval_pts_shapes = self.parameters['eval_pts_shapes']
        eval_pts_location = self.parameters['eval_pts_location']
        delta_t = self.parameters['delta_t']

        submodel = HorseshoeCirculations(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
        )
        self.add(submodel, name='compute_horseshoe_circulation')

        submodel = EvalPtsVel(
            eval_pts_names=eval_pts_names,
            eval_pts_shapes=eval_pts_shapes,
            eval_pts_location=eval_pts_location,
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            nt=nt,
            delta_t=delta_t,
        )
        self.add(submodel, name='EvalPtsVel')

        submodel = LiftDrag(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
        )
        self.add(submodel, name='LiftDrag')


if __name__ == "__main__":

    def generate_simple_mesh(nx, ny, nt=None):
        if nt == None:
            mesh = np.zeros((nx, ny, 3))
            mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))
            mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
            mesh[:, :, 2] = 0.
        else:
            mesh = np.zeros((nt, nx, ny, 3))
            for i in range(nt):
                mesh[i, :, :, 0] = np.outer(np.arange(nx), np.ones(ny))
                mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
                mesh[i, :, :, 2] = 0.
        return mesh

    nx = 3
    ny = 4
    nt = 5
    delta_t = 1
    surface_names = ['wing']
    surface_shapes = [(nx, ny, 3)]
    eval_pts_names = ['eval_pts_coords']
    eval_pts_shapes = [(nx - 1, ny - 1, 3)]

    model_1 = Model()

    model_1.add(
        Outputs(surface_names=surface_names,
                surface_shapes=surface_shapes,
                nt=nt,
                eval_pts_names=eval_pts_names,
                eval_pts_shapes=eval_pts_shapes,
                delta_t=delta_t))

    sim = Simulator(model_1)
    sim.run()
    sim.visualize_implementation()
#     # print('aic is', sim['aic'])
#     # print('v_ind is', sim['v_ind'].shape, sim['v_ind'])