from VLM_package.VLM_system.vlm_system import VLMSystemModel
from VLM_package.VLM_outputs.compute_force.compute_outputs_group import Outputs
import numpy as np

from VLM_package.VLM_preprocessing.generate_simple_mesh import *


class VLMSolverModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('free_stream_velocities', types=np.ndarray)

    def define(self):
        # add the mesh info
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        free_stream_velocities = self.parameters['free_stream_velocities']
        frame_vel_val = -free_stream_velocities

        frame_vel = self.create_input('frame_vel', val=frame_vel_val)

        self.add(
            VLMSystemModel(
                surface_names=surface_names,
                surface_shapes=surface_shapes,
                # frame_vel=frame_vel_val,
            ),
            'ODE_system')

        eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
        eval_pts_shapes = [(x[0] - 1, x[1] - 1, 3) for x in surface_shapes]

        # compute lift and drag
        sub = Outputs(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            eval_pts_names=eval_pts_names,
            eval_pts_shapes=eval_pts_shapes,
        )
        self.add(sub, name='compute_lift_drag')


if __name__ == "__main__":

    nx = 2
    ny = 3
    surface_shapes = [(nx, ny, 3)]

    model = Model()
    mesh_val = generate_simple_mesh(nx, ny).reshape(1, nx, ny, 3)
    surface_names = ['wing']
    frame_vel_val = np.array([-50, 0, -1])
    free_stream_velocities = -frame_vel_val

    wing = model.create_input('wing', val=mesh_val)

    submodel = VLMSolverModel(
        surface_names=surface_names,
        surface_shapes=surface_shapes,
        free_stream_velocities=free_stream_velocities,
    )
    model.add(submodel, 'VLMSolverModel')
    sim = Simulator(model)

    sim.run()
    print('lift', sim.prob['L'])
    print('drag', sim.prob['D'])
    # sim.visualize_implementation()
