from VLM_package.VLM_system.vlm_system import VLMSystemModel
from VLM_package.VLM_outputs.compute_force.compute_outputs_group import Outputs
import numpy as np

from VLM_package.VLM_preprocessing.generate_simple_mesh import *
from VLM_package.VLM_outputs.compute_effective_aoa_cd_v import AOA_CD

# Here nt is just a dummy variable that always equal to 2. since we are using a long wake panel,
# we can just make nt=2 and delta_t a large number.


class VLMSolverModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('num_nodes', types=int)

        # self.parameters.declare('free_stream_velocities', types=np.ndarray)

        self.parameters.declare('eval_pts_location', default=0.25)

        self.parameters.declare('eval_pts_option', default='auto')
        self.parameters.declare('eval_pts_shapes', types=list)
        self.parameters.declare('sprs', default=None)
        self.parameters.declare('model_name')

        self.parameters.declare('coeffs_aoa', default=None)
        self.parameters.declare('coeffs_cd', default=None)

    def define(self):
        # add the mesh info
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        num_nodes = self.parameters['num_nodes']
        model_name = self.parameters['model_name']

        # free_stream_velocities = self.parameters['free_stream_velocities']

        eval_pts_option = self.parameters['eval_pts_option']

        eval_pts_location = self.parameters['eval_pts_location']
        eval_pts_shapes = self.parameters['eval_pts_shapes']
        sprs = self.parameters['sprs']

        coeffs_aoa = self.parameters['coeffs_aoa']
        coeffs_cd = self.parameters['coeffs_cd']

        # frame_vel_val = -free_stream_velocities
        aoa = self.declare_variable(model_name + 'aoa', shape=(num_nodes, 1))
        side_slip_ang = self.create_input(model_name + 'side_slip_ang',
                                          val=np.zeros((num_nodes, 1)))
        v_inf = self.declare_variable(model_name + 'v_inf',
                                      shape=(num_nodes, 1))

        alpha = aoa / 180 * np.pi
        beta = side_slip_ang / 180 * np.pi

        vx = -v_inf * csdl.cos(alpha) * csdl.cos(beta)
        vy = v_inf * csdl.sin(beta)
        vz = -v_inf * csdl.sin(alpha) * csdl.cos(beta)

        frame_vel = self.create_output(model_name + 'frame_vel',
                                       shape=(num_nodes, 3))
        frame_vel[:, 0] = vx
        frame_vel[:, 1] = vy
        frame_vel[:, 2] = vz
        print('vlm_solver frame_vel shape', frame_vel.shape)

        self.add(
            VLMSystemModel(
                surface_names=surface_names,
                surface_shapes=surface_shapes,
                num_nodes=num_nodes,
                model_name=model_name,
                # frame_vel=frame_vel_val,
            ),
            'VLM_system')

        eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]

        # compute lift and drag
        sub = Outputs(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            eval_pts_names=eval_pts_names,
            eval_pts_shapes=eval_pts_shapes,
            eval_pts_option=eval_pts_option,
            eval_pts_location=eval_pts_location,
            sprs=sprs,
            model_name=model_name,
            coeffs_aoa=coeffs_aoa,
            coeffs_cd=coeffs_cd,
        )
        self.add(sub, name='VLM_outputs')

        # coeffs_aoa = [np.loadtxt('cl_aoa_coeff.txt')]
        # coeffs_cd = [np.loadtxt('cd_aoa_coeff.txt')]

        # sub = AOA_CD(
        #     surface_names=surface_names,
        #     surface_shapes=surface_shapes,
        #     coeffs_aoa=coeffs_aoa,
        #     coeffs_cd=coeffs_cd,
        # )
        # self.add(sub, name='AOA_CD')


if __name__ == "__main__":

    pass
