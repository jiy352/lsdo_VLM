import numpy as np

from VLM_package.VLM_preprocessing.generate_simple_mesh import *

from VLM_package.vlm_solver import VLMSolverModel
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
free_stream_velocities = np.array([-vx, 0, -vz])
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

rot_vel = model_1.create_input(surface_names[0] + '_rot_vel',
                               val=np.zeros((nx, ny, 3)))
rot_vel_1 = model_1.create_input(surface_names[1] + '_rot_vel',
                                 val=np.zeros((nx, ny - 1, 3)))

wing = model_1.create_input('wing', val=mesh_val)
wing_1 = model_1.create_input('wing_1', val=mesh_val_1)

submodel = VLMSolverModel(
    surface_names=surface_names,
    surface_shapes=surface_shapes,
    free_stream_velocities=free_stream_velocities,
    eval_pts_location=0.25,
    # The location of the evaluation point is on the quarter-chord,
    # if this is not provided, it is defaulted to be 0.25.
    # Or it can be set to a set of points the user defines. like below:
)
model_1.add(submodel, 'VLMSolverModel')

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
sim.visualize_implementation()
