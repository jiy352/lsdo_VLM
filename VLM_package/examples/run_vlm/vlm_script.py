import matplotlib.pyplot as plt
import openmdao.api as om
from ozone2.api import ODEProblem, Wrap

from VLM_package.VLM_system.vlm_system import ODESystemModel
from VLM_package.VLM_outputs.compute_force.compute_outputs_group import Outputs
import csdl_om
import numpy as np

from VLM_package.VLM_preprocessing.generate_simple_mesh import *

nt = 5
nx = 2
ny = 4
delta_t = h_stepsize = 1
offset = 10

frame_vel_val = np.array([-1, 0, -1])
wake_coords_val = compute_wake_coords(nx, ny, nt, h_stepsize,
                                      frame_vel_val).reshape(1, nt, ny, 3)

wake_coords_val_1 = compute_wake_coords(nx, ny - 1, nt, h_stepsize,
                                        frame_vel_val,
                                        offset).reshape(1, nt, ny - 1, 3)

# surface_names = ['wing', 'wing_1']
# surface_shapes = [(nx, ny, 3), (nx, ny - 1, 3)]
# wake_coords = [wake_coords_val, wake_coords_val_1]

# single lifting surface

surface_names = ['wing']
surface_shapes = [(nx, ny, 3)]
wake_coords = [wake_coords_val]

model_1 = csdl.Model()

frame_vel_val = np.array([-1, 0, -1])

mesh_val = generate_simple_mesh(nx, ny).reshape(1, nx, ny, 3)
mesh_val_1 = generate_simple_mesh(nx, ny - 1,
                                  offset=offset).reshape(1, nx, ny - 1, 3)

wing = model_1.create_input('wing', val=mesh_val)
wing_1 = model_1.create_input('wing_1', val=mesh_val_1)
wing_rot_vel = model_1.create_input('wing_rot_vel',
                                    val=np.zeros(((nx - 1) * (ny - 1), 3)))
wing_rot_vel_1 = model_1.create_input('wing_1_rot_vel',
                                      val=np.zeros(
                                          ((nx - 1) * (ny - 1 - 1), 3)))

# add the mesh info
model_1.add(
    ODESystemModel(surface_names=surface_names,
                   surface_shapes=surface_shapes,
                   frame_vel=frame_vel_val,
                   nt=nt,
                   delta_t=delta_t), 'ODE_system')

eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
eval_pts_shapes = [(x[0] - 1, x[1] - 1, 3) for x in surface_shapes]
# compute lift and drag
sub = Outputs(surface_names=surface_names,
              surface_shapes=surface_shapes,
              nt=nt,
              eval_pts_names=eval_pts_names,
              eval_pts_shapes=eval_pts_shapes,
              delta_t=delta_t)
model_1.add(sub, name='compute_lift_drag')

sim = Simulator(model_1)

sim.run()
print('lift', sim.prob['L'])
print('drag', sim.prob['D'])
sim.visualize_implementation()
