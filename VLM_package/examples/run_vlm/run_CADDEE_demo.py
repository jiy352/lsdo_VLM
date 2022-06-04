from cProfile import label
import numpy as np

from VLM_package.VLM_preprocessing.generate_simple_mesh import *

from VLM_package.vlm_solver import VLMSolverModel

from VLM_package.examples.run_vlm.utils.generate_mesh import generate_mesh

from VLM_package.examples.run_vlm.AcStates_enum import *
# import pyvista as pv
'''
This example demonstrates the basic VLM simulation 
with a single lifting surface with internal function to generate evaluation pts
Please see vlm_scipt_mls.py for how to use user defined evaluation pts
'''

####################################################################
# 1. Define VLM inputs that share the common names within CADDEE
####################################################################
num_nodes = 1
create_opt = 'create_inputs'
model_1 = csdl.Model()

print('creating inputs that share the same names within CADDEE:')

for data in AcStates:
    print('{:15} = {}'.format(data.name, data.value))
    name = data.name
    string_name = data.value
    if create_opt == 'create_inputs':
        variable = model_1.create_input(string_name,
                                        val=AcStates_val_dict[string_name])
        # del variable
    else:
        variable = model_1.declare_variable(string_name,
                                            val=AcStates_val_dict[string_name])
        # del variable

####################################################################
# 2. add VLM meshes
####################################################################
# single lifting surface
nx = 3  # number of points in streamwise direction
ny = 11  # number of points in spanwise direction

surface_names = ['wing', 'wing_1']
surface_shapes = [(num_nodes, nx, ny, 3), (num_nodes, nx, ny, 3)]

chord = 1.49352
span = 16.2 / chord
# https://github.com/LSDOlab/nasa_uli_tc1/blob/222d877228b609076dd352945f4cfe2d158d4973/execution_scripts/c172_climb.py#L33

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

# Generate mesh of a rectangular wing
mesh = generate_mesh(mesh_dict)
offset = span / 2

# mesh_val = generate_simple_mesh(nx, ny, num_nodes)
mesh_val = np.zeros((num_nodes, nx, ny, 3))
mesh_val_1 = np.zeros((num_nodes, nx, ny, 3))

for i in range(num_nodes):
    mesh_val[i, :, :, :] = mesh
    mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0]
    mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1] + offset

    mesh_val_1[i, :, :, :] = mesh.copy()
    mesh_val_1[i, :, :, 1] = mesh.copy()[:, :, 1] - offset
    mesh_val_1[i, :, :, 0] = mesh.copy()[:, :, 0]

wing = model_1.create_input('wing', val=mesh_val)
wing_1 = model_1.create_input('wing_1', val=mesh_val_1)
# '''
############################################
# Plot the lifting surfaces
############################################
# pv.global_theme.axes.show = True
# pv.global_theme.font.label_size = 1
# x = mesh_val[0, :, :, 0]
# y = mesh_val[0, :, :, 1]
# z = mesh_val[0, :, :, 2]
# x_1 = mesh_val_1[0, :, :, 0]
# y_1 = mesh_val_1[0, :, :, 1]
# z_1 = mesh_val_1[0, :, :, 2]

# grid = pv.StructuredGrid(x, y, z)
# grid_1 = pv.StructuredGrid(x_1, y_1, z_1)

# grid.cell_data["panel_forces"] = np.moveaxis(
#     sim["panel_forces"][0, :56, :].reshape(nx - 1, ny - 1, 3), 0,
#     1).reshape(56, 3)
# grid_1.cell_data["panel_forces"] = np.moveaxis(
#     sim["panel_forces"][0, 56:, :].reshape(nx - 1, ny - 1, 3), 0,
#     1).reshape(56, 3)
# grid.save("vtks/left_wing.vtk")
# grid_1.save("vtks/right_wing.vtk")
# p = pv.Plotter()
# p.add_mesh(grid, color="blue", show_edges=True, opacity=.5)
# p.add_mesh(grid_1, color="red", show_edges=True, opacity=.5)
# p.camera.view_angle = 60.0
# p.add_axes_at_origin(labels_off=True, line_width=5)

# p.show()
# exit()

####################################################################
# 2. preprocessing to connect to the vlm solver
####################################################################

# rot_vel = model_1.create_input(surface_names[0] + '_rot_vel',
#                                val=np.zeros((num_nodes, nx, ny, 3)))

# v_inf = model_1.create_input('v_inf', val=v_inf.reshape(-1, 1))

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
    # free_stream_velocities=free_stream_velocities,
    eval_pts_location=0.25,
    # The location of the evaluation point is on the quarter-chord,
    # if this is not provided, it is defaulted to be 0.25.
    eval_pts_shapes=eval_pts_shapes,
    AcStates=AcStates,
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
