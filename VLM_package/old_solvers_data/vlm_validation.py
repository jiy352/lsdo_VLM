from filecmp import clear_cache
from VLM_package.VLM_system.vlm_system import VLMSystemModel
from VLM_package.VLM_outputs.compute_force.compute_outputs_group import Outputs
import numpy as np

from VLM_package.VLM_preprocessing.generate_simple_mesh import *
from VLM_package.VLM_outputs.compute_effective_aoa_cd_v import AOA_CD

from openaerostruct.geometry.utils import generate_mesh

from click import clear


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

        coeffs_aoa = [np.loadtxt('cl_aoa_coeff.txt')]
        coeffs_cd = [np.loadtxt('cd_aoa_coeff.txt')]

        sub = AOA_CD(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            coeffs_aoa=coeffs_aoa,
            coeffs_cd=coeffs_cd,
        )
        self.add(sub, name='AOA_CD')


if __name__ == "__main__":

    CL_list = []
    CD_list = []
    __saved_context__ = {}
    for span in [6, 8, 10, 12, 14]:
        for alpha_deg in [0, 2, 4, 6, 8, 10, 12, 14]:
            clear_cache()
            nx = 3
            ny = 5
            chord = 1
            surface_shapes = [(nx, ny, 3)]

            v_inf = 50
            alpha = alpha_deg / 180 * np.pi
            vx = -v_inf * np.cos(alpha)
            vz = -v_inf * np.sin(alpha)

            mesh_dict = {
                "num_y": ny,
                "num_x": nx,
                "wing_type": "rect",
                "symmetry": False,
                "span": span,
                "chord": chord,
                "span_cos_spacing": False,
                "chord_cos_spacing": False,
            }

            # Generate half-wing mesh of rectangular wing
            mesh = generate_mesh(mesh_dict)

            # vx = 50
            # vz = 5
            # frame_vel_val = np.array([vx, 0, vz])

            model = Model()
            # mesh_val = generate_simple_mesh(nx, ny).reshape(1, nx, ny, 3)
            # mesh_val = np.loadtxt('points.txt').reshape(nx, ny + 1, 3)[:, :-1, :]
            # vp_init = Plotter()
            # vps1 = Points(mesh_val.reshape(nx * ny, 3), r=8, c='blue')
            # vp_init.show(vps1, 'Camber', axes=1, viewup="z", interactive=True)
            surface_names = ['wing']
            free_stream_velocities = np.array([-vx, 0, -vz])

            # mesh = rearranged_arr = np.moveaxis(mesh_val, [0, 1], [1, 0])

            wing = model.create_input('wing', val=mesh.reshape(1, nx, ny, 3))

            submodel = VLMSolverModel(
                surface_names=surface_names,
                surface_shapes=surface_shapes,
                free_stream_velocities=free_stream_velocities,
            )
            model.add(submodel, 'VLMSolverModel')
            sim = Simulator(model)

            sim.run()
            CL_list.append(float(sim.prob['C_L']))
            CD_list.append(float(sim.prob['C_D_i']))
            clear_cache()
            print(sim['L'])
            del sim
            del model
            del submodel
            del wing
            del mesh

    print('CL_list', CL_list)
    print('CD_list', CD_list)
    np.savetxt('CL_list.out', CL_list)
    np.savetxt('CD_list.out', CD_list)
