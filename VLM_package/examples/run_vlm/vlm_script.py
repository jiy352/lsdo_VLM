from VLM_package.VLM_system.vlm_system import VLMSystemModel
from VLM_package.VLM_outputs.compute_force.compute_outputs_group import Outputs
import numpy as np

from VLM_package.VLM_preprocessing.generate_simple_mesh import *

# here nt is just a dummy variable that always equal to 2. since we are using a long wake panel,
# we can just make nt=2, delta_t=a large number.

nx = 2
ny = 4
offset = 10

frame_vel_val = np.array([1e-9, 0, -1])

# multiple lifting surface
surface_names = ['wing', 'wing_1']
surface_shapes = [(nx, ny, 3), (nx, ny - 1, 3)]

# single lifting surface
# surface_names = ['wing']
# surface_shapes = [(nx, ny, 3)]

model_1 = csdl.Model()

mesh_val = generate_simple_mesh(nx, ny).reshape(1, nx, ny, 3)
mesh_val_1 = generate_simple_mesh(nx, ny - 1,
                                  offset=offset).reshape(1, nx, ny - 1, 3)

wing = model_1.create_input('wing', val=mesh_val)
wing_1 = model_1.create_input('wing_1', val=mesh_val_1)

# add the mesh info
model_1.add(
    VLMSystemModel(
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
