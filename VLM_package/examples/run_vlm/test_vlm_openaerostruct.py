import matplotlib.pyplot as plt

from VLM_package.VLM_system.vlm_system import VLMSystemModel
from VLM_package.VLM_outputs.compute_force.compute_outputs_group import Outputs
import numpy as np

from openaerostruct_generate_mesh import generate_mesh

from VLM_package.VLM_preprocessing.generate_simple_mesh import *

nt = 2

num_x = 3
num_y = 5
span = 20.
chord = 1.

offset = 10

mesh_dict = {
    "num_y": num_y,
    "num_x": num_x,
    "wing_type": "rect",
    "symmetry": False,
    "span": span,
    "chord": chord,
    "span_cos_spacing": False,
    "chord_cos_spacing": False,
}

# Generate half-wing mesh of rectangular wing
mesh = generate_mesh(mesh_dict)
# mesh = generate_mesh(mesh_dict)
plt.plot(mesh[:, :, 0], mesh[:, :, 1], '.')
plt.show()

v_inf = 50

alpha_deg = 10
alpha = alpha_deg / 180 * np.pi
vx = -v_inf * np.cos(alpha)
vz = -v_inf * np.sin(alpha)

frame_vel_val = np.array([vx, 0, vz])

# single lifting surface
surface_names = ['wing']
surface_shapes = [(num_x, num_y, 3)]

model_1 = csdl.Model()

wing = model_1.create_input('wing', val=mesh.reshape(1, num_x, num_y, 3))
frame_vel = model_1.create_input('frame_vel', val=frame_vel_val)

# add the mesh info
model_1.add(
    VLMSystemModel(
        surface_names=surface_names,
        surface_shapes=surface_shapes,
    ), 'ODE_system')

eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
eval_pts_shapes = [(x[0] - 1, x[1] - 1, 3) for x in surface_shapes]
# compute lift and drag
sub = Outputs(surface_names=surface_names,
              surface_shapes=surface_shapes,
              nt=nt,
              eval_pts_names=eval_pts_names,
              eval_pts_shapes=eval_pts_shapes)
model_1.add(sub, name='compute_lift_drag')

sim = Simulator(model_1)

sim.run()
print('lift', sim.prob['L'])
print('drag', sim.prob['D'])
sim.visualize_implementation()
# sim['aic_bd_proj'] @ sim['gamma_b'] + sim['M'] @ sim['gamma_w'].reshape(
#     num_y - 1) + sim['b']

# sim['aic_bd_proj'] @ sim['gamma_b'] + sim['M'] @ np.ones(num_y - 1) + sim['b']
