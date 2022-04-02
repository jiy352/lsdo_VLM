from VLM_package.VLM_system.vlm_system import VLMSystemModel
from VLM_package.VLM_outputs.compute_force.compute_outputs_group import Outputs
import numpy as np

from VLM_package.VLM_preprocessing.generate_simple_mesh import *

# here nt is just a dummy variable that always equal to 2. since we are using a long wake panel,
# we can just make nt=2, delta_t=a large number.

nx = 2
ny = 20
# ny = 100
offset = 10

v_inf = 50
alpha_deg = 10
alpha = alpha_deg / 180 * np.pi
vx = -v_inf * np.cos(alpha)
vz = -v_inf * np.sin(alpha)

# vx = 50
# vz = 5
frame_vel_val = np.array([vx, 0, vz])

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

frame_vel = model_1.create_input('frame_vel', val=frame_vel_val)

wing = model_1.create_input('wing', val=mesh_val)
wing_1 = model_1.create_input('wing_1', val=mesh_val_1)

# add the mesh info
model_1.add(
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
model_1.add(sub, name='compute_lift_drag')

sim = Simulator(model_1)

sim.run()
for i in range(len(surface_names)):

    L_panel_name = surface_names[i] + '_L_panel'
    D_panel_name = surface_names[i] + '_D_panel'
    L_name = surface_names[i] + '_L'
    D_name = surface_names[i] + '_D'
    CL_name = surface_names[i] + '_C_L'
    CD_name = surface_names[i] + '_C_D_i'
    print('lift', L_name, sim.prob[L_name])
    print('drag', D_name, sim.prob[D_name])
    print(
        'L_panel',
        L_panel_name,
        sim.prob[L_panel_name].shape,
        sim.prob[L_panel_name],
    )
    print(
        'D_panel',
        D_panel_name,
        sim.prob[D_panel_name].shape,
        sim.prob[D_panel_name],
    )
    print('cl', CL_name, sim.prob[CL_name])
    print('cd', CD_name, sim.prob[CD_name])
# sim.visualize_implementation()
