from VLM_package.VLM_system.vlm_system import VLMSystem
from VLM_package.VLM_outputs.compute_force.compute_outputs_group import Outputs
import numpy as np

from VLM_package.VLM_preprocessing.utils.generate_simple_mesh import *

from vedo import *
# here n_wake_pts_chord is just a dummy variable that always equal to 2. since we are using a long wake panel,
# we can just make n_wake_pts_chord=2, delta_t=a large number.

nx = 3
ny = 20

frame_vel_val = np.array([-1, 0, 0])

# single lifting surface
surface_names = ['wing']
surface_shapes = [(nx, ny, 3)]

model_1 = csdl.Model()
mesh_org = np.loadtxt('test.txt').reshape(nx, ny, 3)
mesh_val = rearranged_arr = np.moveaxis(mesh_org, [0, 1], [1, 0])

vp_init = Plotter()
vps1 = Points(mesh_val.reshape(nx * ny, 3), r=8, c='blue')
vp_init.show(vps1, 'Camber', axes=1, viewup="z", interactive=True)

wing = model_1.create_input('wing', val=mesh_val.reshape(1, nx, ny, 3))

# add the mesh info
model_1.add(
    VLMSystem(
        surface_names=surface_names,
        surface_shapes=surface_shapes,
        frame_vel=frame_vel_val,
    ), 'ODE_system')

eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
eval_pts_shapes = [(x[0] - 1, x[1] - 1, 3) for x in surface_shapes]
# compute lift and drag
sub = Outputs(
    surface_names=surface_names,
    surface_shapes=surface_shapes,
    eval_pts_names=eval_pts_names,
    eval_pts_shapes=eval_pts_shapes,
)
model_1.add(sub, name='compute_lift_drag')

sim = Simulator(model_1)

sim.run()
print('lift', sim.prob['L'])
print('drag', sim.prob['D'])
sim.visualize_implementation()
