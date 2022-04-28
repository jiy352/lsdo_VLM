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
num_nodes = 4
nx = 3 # streamwise vertices
ny = 5  # chordwise vertices

######################
# v_inf degree
# rho
# aoa
######################

v_inf_val = np.array([50, 60, 70,80]).reshape(-1,1)
aoa_val = np.array([2, 4, 6,8]).reshape(-1,1)
rho_val =  np.array([0.38, 0.40, 0.42,0.44]).reshape(-1,1)

for span in [6]:


    chord = 1  # chord length
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
mesh_val = np.zeros((num_nodes, nx, ny, 3))
mesh_val_1 = np.zeros((num_nodes, nx, ny, 3))
for i in range(num_nodes):
    mesh_val[i, :, :, :] = mesh

mesh_all = [mesh_val]



# single lifting surface
surface_names = ['wing']
surface_shapes = [(num_nodes, nx, ny, 3)]



####################################################################
# 2. Define CSDL inputs
####################################################################

model_1 = csdl.Model()


wing = model_1.create_input('wing', val=mesh_val)
v_inf = model_1.create_input('v_inf', val=v_inf_val)
aoa = model_1.create_input('aoa', val=aoa_val)

rho = model_1.create_input('rho', val=rho_val)

####################################################################
# 2. Add vlm solver model
####################################################################
eval_pts_shapes = [(num_nodes, x[1] - 1, x[2] - 1, 3) for x in surface_shapes]

submodel = VLMSolverModel(
    surface_names=surface_names,
    surface_shapes=surface_shapes,
    num_nodes=num_nodes,
    # free_stream_velocities=free_stream_velocities,
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

