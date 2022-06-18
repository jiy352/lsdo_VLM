from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np


class AdapterComp(Model):
    """
    Compute various geometric properties for VLM analysis.
    These are used primarily to help compute postprocessing quantities,
    such as wave CD, viscous CD, etc.
    Some of the quantities, like `normals`, are used to compute the RHS
    of the AIC linear system.

    parameters
    ----------
    def_mesh[nx, ny, 3] : numpy array
        Array defining the nodal coordinates of the lifting surface.

    Returns
    -------
    # b_pts[nx, ny, 3] : csdl array
    #     Bound points for the horseshoe vortices, found along the 1/4 chord.
    bd_vtx_coords[nx, ny, 3] : csdl array
    bound vortices points
    coll_pts[nx-1, ny-1, 3] : csdl array
        collocation points for the horseshoe vortices, found along the 3/4 chord.
    l_span[(nx-1)* (ny-1)] : csdl array
        The spanwise widths of each individual panel.
    l_chord[(nx-1)* (ny-1)] : csdl array
        The chordwise length of of each individual panel.
    s_panel [(nx-1)* (ny-1)]: csdl array
        The panel areas.
    """
    def initialize(self):
        self.parameters.declare('AcStates')
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)

    def define(self):
        # add_input
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']

        num_nodes = surface_shapes[0][0]

        AcStates = self.parameters['AcStates']

        u = self.declare_variable(AcStates.u.value, shape=(num_nodes, 1))
        v = self.declare_variable(AcStates.v.value, shape=(num_nodes, 1))
        w = self.declare_variable(AcStates.w.value, shape=(num_nodes, 1))

        # we always assume v_inf > 0 here
        v_inf = (u**2 + v**2 + w**2)**0.5
        self.register_output('v_inf', v_inf)

        # print('u shape', u.shape)

        frame_vel = self.create_output('frame_vel', shape=(num_nodes, 3))
        # print('frame_vel[:, 0] shape', frame_vel[:, 0].shape)

        #####################################
        # TODO fix when u>0
        #####################################

        frame_vel[:, 0] = u
        frame_vel[:, 1] = v
        frame_vel[:, 2] = w

        alpha = csdl.arctan(frame_vel[:, 2] / frame_vel[:, 0])

        sinbeta = csdl.reshape(frame_vel[:, 1] / v_inf,
                               new_shape=(num_nodes, ))

        beta = csdl.reshape(-csdl.arcsin(sinbeta), new_shape=(num_nodes, 1))

        self.register_output('alpha', alpha)
        self.register_output('beta', beta)

        p = self.declare_variable(AcStates.p.value, shape=(num_nodes, 1))
        q = self.declare_variable(AcStates.q.value, shape=(num_nodes, 1))
        r = self.declare_variable(AcStates.r.value, shape=(num_nodes, 1))
        ang_vel = self.create_output('ang_vel', shape=(num_nodes, 3))
        ang_vel[:, 0] = p
        ang_vel[:, 1] = q
        ang_vel[:, 2] = r

        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            nx = surface_shapes[i][1]
            ny = surface_shapes[i][2]

            coll_pts_coords_name = surface_name + '_coll_pts_coords'
            rotatonal_vel_names = surface_name + '_rot_vel_coll'

            coll_pts = self.declare_variable(coll_pts_coords_name,
                                             shape=(num_nodes, nx - 1, ny - 1,
                                                    3))
            r_vec = coll_pts - 0.
            ang_vel_exp = csdl.expand(ang_vel, (num_nodes, nx - 1, ny - 1, 3),
                                      indices='il->ijkl')
            rot_vel = csdl.cross(ang_vel_exp, r_vec, axis=3)
            self.register_output(rotatonal_vel_names, rot_vel)


if __name__ == "__main__":
    pass