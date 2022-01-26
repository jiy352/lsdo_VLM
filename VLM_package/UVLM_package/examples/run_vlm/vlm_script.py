import matplotlib.pyplot as plt
import openmdao.api as om
from ozone2.api import ODEProblem, Wrap

from VLM_package.VLM_system.ode_system import ODESystemModel
from VLM_package.VLM_outputs.compute_force.compute_outputs_group import Outputs
import csdl_om
import numpy as np

from VLM_package.VLM_preprocessing.generate_simple_mesh import *

nt = 20
nx = 2
ny = 4
nx_1 = 2
ny_1 = 4

delta_t = h_stepsize = 1
offset = 10

frame_vel_val = np.array([-1, 0, -1])
wake_coords_val = compute_wake_coords(nx, ny, nt, h_stepsize,
                                      frame_vel_val).reshape(1, nt, ny, 3)

wake_coords_val_1 = compute_wake_coords(nx_1, ny_1, nt, h_stepsize,
                                        frame_vel_val,
                                        offset).reshape(1, nt, ny_1, 3)

surface_names = ['wing']
surface_shapes = [(nx, ny, 3)]
wake_coords = [wake_coords_val]

# ##################uncomment this for multiple lifting surfaces:###################
# surface_names = ['wing', 'wing_1']
# shape_1 = (nx, ny, 3)
# shape_2 = (nx_1, ny_1, 3)
# surface_shapes = [shape_1, shape_2]
# wake_coords = [wake_coords_val, wake_coords_val_1]

model_1 = csdl.Model()

frame_vel_val = np.array([-1, 0, -1])

mesh_val = generate_simple_mesh(nx, ny).reshape(1, nx, ny, 3)
mesh_val_1 = generate_simple_mesh(nx_1, ny_1,
                                  offset=offset).reshape(1, nx_1, ny_1, 3)

wing = model_1.create_input('wing', val=mesh_val)
wing_1 = model_1.create_input('wing_1', val=mesh_val_1)
wing_rot_vel = model_1.create_input('wing_rot_vel',
                                    val=np.zeros(((nx - 1) * (ny - 1), 3)))
wing_rot_vel_1 = model_1.create_input('wing_1_rot_vel',
                                      val=np.zeros(
                                          ((nx_1 - 1) * (ny_1 - 1), 3)))

# add the mesh info
model_1.add(
    ODESystemModel(surface_names=surface_names,
                   surface_shapes=surface_shapes,
                   frame_vel=frame_vel_val,
                   wake_coords=wake_coords,
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
