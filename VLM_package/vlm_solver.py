from VLM_package.VLM_system.vlm_system import VLMSystem
from VLM_package.VLM_outputs.compute_force.compute_outputs_group import Outputs
import numpy as np

from VLM_package.VLM_preprocessing.utils.generate_simple_mesh import *

# Here n_wake_pts_chord is just a dummy variable that always equal to 2. since we are using a long wake panel,
# we can just make n_wake_pts_chord=2 and delta_t a large number.


class VLMSolverModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('num_nodes', types=int)

        self.parameters.declare('AcStates', default=None)

        self.parameters.declare('free_stream_velocities', default=None)

        self.parameters.declare('eval_pts_location', default=0)
        self.parameters.declare('eval_pts_names', default=None)

        self.parameters.declare('eval_pts_option', default='auto')
        self.parameters.declare('eval_pts_shapes', types=list)
        self.parameters.declare('sprs', default=None)
        self.parameters.declare('coeffs_aoa', default=None)
        self.parameters.declare('coeffs_cd', default=None)
        self.parameters.declare('solve_option', default='direct')
        self.parameters.declare('TE_idx', default='last')
        self.parameters.declare('mesh_unit', default='m')
        self.parameters.declare('cl0', default=[0])

    def define(self):
        # add the mesh info
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = self.parameters['num_nodes']
        cl0 = self.parameters['cl0']

        free_stream_velocities = self.parameters['free_stream_velocities']

        eval_pts_option = self.parameters['eval_pts_option']

        eval_pts_location = self.parameters['eval_pts_location']
        eval_pts_shapes = self.parameters['eval_pts_shapes']
        sprs = self.parameters['sprs']

        coeffs_aoa = self.parameters['coeffs_aoa']
        coeffs_cd = self.parameters['coeffs_cd']
        mesh_unit = self.parameters['mesh_unit']
        if self.parameters['AcStates'] == None:
            frame_vel_val = -free_stream_velocities

            frame_vel = self.create_input('frame_vel', val=frame_vel_val)

        self.add(
            VLMSystem(
                surface_names=surface_names,
                surface_shapes=surface_shapes,
                num_nodes=num_nodes,
                AcStates=self.parameters['AcStates'],
                solve_option=self.parameters['solve_option'],
                TE_idx=self.parameters['TE_idx'],
                mesh_unit=mesh_unit,
            ), 'VLM_system')
        if eval_pts_option=='auto':
            eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
        else:
            eval_pts_names=self.parameters['eval_pts_names']

        # compute lift and drag
        sub = Outputs(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            eval_pts_names=eval_pts_names,
            eval_pts_shapes=eval_pts_shapes,
            eval_pts_option=eval_pts_option,
            eval_pts_location=eval_pts_location,
            sprs=sprs,
            coeffs_aoa=coeffs_aoa,
            coeffs_cd=coeffs_cd,
            mesh_unit=mesh_unit,
            cl0=cl0,
        )
        self.add(sub, name='VLM_outputs')


if __name__ == "__main__":

    pass
