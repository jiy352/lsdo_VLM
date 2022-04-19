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
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        # get num_nodes from surface shape
        num_nodes = surface_shapes[0][0]

        # add_input name and shapes
        rotatonal_vel_names = [x + '_rot_vel' for x in surface_names]
        # compute rotatonal_vel_shapes from surface shape
        bd_coll_pts_shapes = [
            tuple(map(lambda i, j: i - j, item, (0, 1, 1, 0)))
            for item in surface_shapes
        ]

        # print('line 49 bd_coll_pts_shapes', bd_coll_pts_shapes)

        rotatonal_vel_shapes = surface_shapes
        # add_output name and shapes
        kinematic_vel_names = [
            x + '_kinematic_vel' for x in self.parameters['surface_names']
        ]

        frame_vel = self.declare_variable('frame_vel', shape=(num_nodes, 3))

        for i in range(len(rotatonal_vel_names)):
            rotatonal_vel_name = rotatonal_vel_names[i]
            rotatonal_vel_shape = rotatonal_vel_shapes[i]
            kinematic_vel_name = kinematic_vel_names[i]

            # TODO: fix this to use actual rotational vel
            # this is fixed
            rotatonal_vel_vertices = self.declare_variable(
                rotatonal_vel_name, val=np.zeros(rotatonal_vel_shape))

            # print('line 69 rotatonal_vel_vertices',
            #       rotatonal_vel_vertices.shape)
            nx = rotatonal_vel_shape[1]
            ny = rotatonal_vel_shape[2]
            rotatonal_vel = csdl.reshape(
                0.25 * (rotatonal_vel_vertices[:, 0:nx - 1, 0:ny - 1, :] +
                        rotatonal_vel_vertices[:, 0:nx - 1, 1:ny, :] +
                        rotatonal_vel_vertices[:, 1:, 0:ny - 1, :] +
                        rotatonal_vel_vertices[:, 1:, 1:, :]),
                new_shape=(num_nodes, (nx - 1) * (ny - 1), 3))
            frame_vel_expand = csdl.expand(frame_vel,
                                           (num_nodes, (nx - 1) * (ny - 1), 3),
                                           indices='ij->ikj')

            # print('line 80 frame_vel_expand shape', frame_vel_expand.shape)
            # print('line 81 rotatonal_vel shape', rotatonal_vel.shape)

            kinematic_vel = -(rotatonal_vel + frame_vel_expand)
            self.register_output(kinematic_vel_name, kinematic_vel)


if __name__ == "__main__":

    surface_names = ['r1', 'r2']
    surface_shapes = [(2, 10, 3, 3), (2, 5, 3, 3)]
    kinematic_vel_names = ['k1', 'k2']
    rotatonal_vel_names = [x + '_rot_vel' for x in surface_names]

    model_1 = Model()
    frame_vel_val = np.random.random((2, 3))

    rot_vel = model_1.create_input(rotatonal_vel_names[0],
                                   val=np.zeros(surface_shapes[0]))
    rot_vel = model_1.create_input(rotatonal_vel_names[1],
                                   val=np.zeros(surface_shapes[1]))

    frame_vel = model_1.create_input('frame_vel', val=frame_vel_val)
    model_1.add(KinematicVelocity(
        surface_names=surface_names,
        surface_shapes=surface_shapes,
    ),
                name='KinematicVelocity')
    sim = Simulator(model_1)
    sim.run()
