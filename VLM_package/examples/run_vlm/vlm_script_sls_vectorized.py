import numpy as np

from VLM_package.VLM_preprocessing.generate_simple_mesh import *

from VLM_package.vlm_solver import VLMSolverModel

from VLM_package.examples.run_vlm.utils.generate_mesh import generate_mesh
'''
This example demonstrates the basic VLM simulation 
with a single lifting surface with internal function to generate evaluation pts
Please see vlm_scipt_mls.py for how to use user defined evaluation pts
'''

####################################################################
# 1. Define VLM meshes and constants
####################################################################

for span in [6]:

    nx = 3
    ny = 5
    chord = 1
    surface_shapes = [(nx, ny, 3)]

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

num_nodes = 3
nx = 3
ny = 5
offset = 10

v_inf = np.array([50, 50, 50])
alpha_deg = np.array([2, 4, 6])
beta_deg = np.array([0, 4, 6])
alpha = alpha_deg / 180 * np.pi
beta = beta_deg / 180 * np.pi
vx = -v_inf * np.cos(alpha) * np.cos(beta)
vy = v_inf * np.sin(beta)
vz = -v_inf * np.sin(alpha) * np.cos(beta)
free_stream_velocities = np.array([-vx, -vy, -vz]).T
frame_vel_val = np.array([vx, vy, vz]).T

# single lifting surface
surface_names = ['wing']
surface_shapes = [(num_nodes, nx, ny, 3)]

model_1 = csdl.Model()

# mesh_val = generate_simple_mesh(nx, ny, num_nodes)
mesh_val = np.zeros((num_nodes, nx, ny, 3))
for i in range(num_nodes):
    mesh_val[i, :, :, :] = mesh

# mesh_val_1 = generate_simple_mesh(nx, ny - 1,
#                                   offset=offset).reshape(1, nx, ny - 1, 3)

mesh_all = [mesh_val]
####################################################################
# 2. Define rotational velocities
# (naming conventions: name=surface_name+'_rot_vel' )
# (you can skip this part if there is no rotation,
# the rotational vel are defaulted to be zeros)
####################################################################

rot_vel = model_1.create_input(surface_names[0] + '_rot_vel',
                               val=np.zeros((num_nodes, nx, ny, 3)))

wing = model_1.create_input('wing', val=mesh_val)
v_inf = model_1.create_input('v_inf', val=v_inf.reshape(-1, 1))
# ##################################################################
# 3. Define VLMSolverModel (using internal function)
# The user needs to provide:
#   surface_names(list),
#   surface_shapes(list),
#   free_stream_velocities(np.array, shape=(3,))
#   eval_pts_location(float)
#   eval_pts_shapes(list)
# Here, the evaluation points are based on the relative
# chordwise panel location generated by the vlm code internally
# This is the default option
# (eval_pts_location=0.25->evaluate the pressure at quarter-chord)
# ###################################################################
# The user can also define the eval_pts_coords inputs (line 97-146)
# ###################################################################

eval_pts_shapes = [(num_nodes, x[1] - 1, x[2] - 1, 3) for x in surface_shapes]

submodel = VLMSolverModel(
    surface_names=surface_names,
    surface_shapes=surface_shapes,
    num_nodes=num_nodes,
    free_stream_velocities=free_stream_velocities,
    eval_pts_location=0.25,
    # The location of the evaluation point is on the quarter-chord,
    # if this is not provided, it is defaulted to be 0.25.
    eval_pts_shapes=eval_pts_shapes,
)

model_1.add(submodel, 'VLMSolverModel')

sim = Simulator(model_1)

sim.run()

####################################################################
# Print VLM outputs
####################################################################

for i in range(len(surface_names)):

    L_panel_name = surface_names[i] + '_L_panel'
    D_panel_name = surface_names[i] + '_D_panel'
    L_name = surface_names[i] + '_L'
    D_name = surface_names[i] + '_D'
    CL_name = surface_names[i] + '_C_L'
    CD_name = surface_names[i] + '_C_D_i'
    print('lift\n', L_name, sim.prob[L_name])
    print('drag\n', D_name, sim.prob[D_name])
    # print(
    #     'L_panel',
    #     L_panel_name,
    #     sim.prob[L_panel_name].shape,
    #     sim.prob[L_panel_name],
    # )
    # print(
    #     'D_panel',
    #     D_panel_name,
    #     sim.prob[D_panel_name].shape,
    #     sim.prob[D_panel_name],
    # )
    print('cl\n', CL_name, sim.prob[CL_name])
    print('cd\n', CD_name, sim.prob[CD_name])
####################################################################
# Visualize n2 diagram (line 188)
####################################################################

# sim.visualize_implementation()
# res = np.einsum('ijk,ik->ij', sim['MTX'], sim['gamma_b']) + sim['b']
# norm = np.linalg.norm(res)
