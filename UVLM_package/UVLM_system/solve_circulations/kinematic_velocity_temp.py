from csdl_om import Simulator
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


class KinematicVelocity(Model):
    """
    Compute various geometric properties for VLM analysis.
    These are used primarily to help compute postprocessing quantities,
    such as wave CD, viscous CD, etc.
    Some of the quantities, like `normals`, are used to compute the RHS
    of the AIC linear system.

    parameters
    ----------
    
    delta_t
    num_t
    axis[str]

    angular_vel[1,] rad/sec
    bd_vtx_coords[num_evel_pts_x, num_evel_pts_y, 3] : csdl array 

    Returns
    -------
    kinematic_vel[nt, num_evel_pts_x, num_evel_pts_y, 3] : csdl array
        Induced velocities at found along the 3/4 chord.
    """
    def initialize(self):

        self.parameters.declare('rotatonal_vel_names', types=list)
        self.parameters.declare('rotatonal_vel_shapes', types=list)
        self.parameters.declare('kinematic_vel_names', types=list)

    def define(self):
        # add_input
        rotatonal_vel_names = self.parameters['rotatonal_vel_names']
        rotatonal_vel_shapes = self.parameters['rotatonal_vel_shapes']
        kinematic_vel_names = self.parameters['kinematic_vel_names']
        frame_vel = self.declare_variable('frame_vel', shape=(3, ))
        for i in range(len(rotatonal_vel_names)):
            rotatonal_vel_name = rotatonal_vel_names[i]
            rotatonal_vel_shape = rotatonal_vel_shapes[i]
            kinematic_vel_name = kinematic_vel_names[i]

            # TODO: fix this to use actual rotational vel
            rotatonal_vel = self.declare_variable(
                rotatonal_vel_name, val=np.zeros(rotatonal_vel_shape))
            frame_vel_expand = csdl.expand(frame_vel,
                                           rotatonal_vel_shape,
                                           indices='i->ji')
            kinematic_vel = -(rotatonal_vel + frame_vel_expand)
            self.register_output(kinematic_vel_name, kinematic_vel)


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

        # self.parameters.declare('rotatonal_vel_names', types=list)
        # self.parameters.declare('rotatonal_vel_shapes', types=list)
        # self.parameters.declare('kinematic_vel_names', types=list)

    rotatonal_vel_names = ['r1', 'r2']
    rotatonal_vel_shapes = [(10, 3), (5, 3)]
    kinematic_vel_names = ['k1', 'k2']

    model_1 = Model()
    frame_vel_val = np.random.random((3, ))

    frame_vel = model_1.create_input('frame_vel', val=frame_vel_val)
    model_1.add(KinematicVelocity(
        rotatonal_vel_names=rotatonal_vel_names,
        rotatonal_vel_shapes=rotatonal_vel_shapes,
        kinematic_vel_names=kinematic_vel_names,
    ),
                name='KinematicVelocity')
    sim = Simulator(model_1)
    sim.run()
    # print('aic is', sim['aic'])
    # print('v_ind is', sim['v_ind'].shape, sim['v_ind'])
