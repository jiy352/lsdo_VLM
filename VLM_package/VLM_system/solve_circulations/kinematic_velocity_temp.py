from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size


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

        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)

    def define(self):
        # add_input name and shapes
        rotatonal_vel_names = [
            x + '_rot_vel' for x in self.parameters['surface_names']
        ]
        # compute rotatonal_vel_shapes from surface shape
        surface_shapes = self.parameters['surface_shapes']
        bd_coll_pts_shapes = [
            tuple(map(lambda i, j: i - j, item, (1, 1, 0)))
            for item in surface_shapes
        ]
        rotatonal_vel_shapes = [
            tuple((item[0] * item[1], item[2])) for item in bd_coll_pts_shapes
        ]
        # add_output name and shapes
        kinematic_vel_names = [
            x + '_kinematic_vel' for x in self.parameters['surface_names']
        ]

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
