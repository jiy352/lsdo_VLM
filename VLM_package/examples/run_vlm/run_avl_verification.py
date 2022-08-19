# import csdl_lite
# from VLM_package.examples.run_vlm.AcStates_enum_vlm import AcStates_vlm
from numpy import indices
import numpy as np

from VLM_package.VLM_preprocessing.utils.generate_simple_mesh import *

from VLM_package.vlm_solver import VLMSolverModel

from VLM_package.examples.run_vlm.utils.generate_mesh import generate_mesh
import enum
from csdl import GraphRepresentation
from csdl_om import Simulator
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

class AcStates_vlm(enum.Enum):
    u = 'u'
    v = 'v'
    w = 'w'
    p = 'p'
    q = 'q'
    r = 'r'
    phi = 'phi'
    theta = 'theta'
    psi = 'psi'
    x = 'x'
    y = 'y'
    z = 'z'
    phiw = 'phiw'
    gamma = 'gamma'
    psiw = 'psiw'
    # rho = 'rho'



# v_inf = np.array([50])
# alpha_deg = np.linspace(-10,-7.5, num_nodes).reshape(num_nodes, 1)
alpha_deg = np.array([10]).reshape(num_nodes, 1)
alpha = alpha_deg / 180 * np.pi
# vx = -v_inf * np.cos(alpha)
# vz = -v_inf * np.sin(alpha)

vx = np.ones((num_nodes,1))*50
vz = np.ones((num_nodes,1))*0
# vx**2+vz**2=v_inf

AcStates_val_dict = {
    AcStates_vlm.u.value: vx.reshape(num_nodes, 1),
    AcStates_vlm.v.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.w.value: vz.reshape(num_nodes, 1),
    AcStates_vlm.p.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.q.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.r.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.phi.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.theta.value: np.ones((num_nodes, 1))*alpha,
    AcStates_vlm.psi.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.x.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.y.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.z.value: np.ones((num_nodes, 1))*0,
    AcStates_vlm.phiw.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.gamma.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.psiw.value: np.zeros((num_nodes, 1)),
    # AcStates_vlm.rho.value: np.ones((num_nodes, 1)) * 0.96,
}



for data in AcStates_vlm:
    print('{:15} = {}={}'.format(data.name, data.value,AcStates_val_dict[data.value]))
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
nx = 7  # number of points in streamwise direction
ny = 13  # number of points in spanwise direction

# surface_names = ['wing', 'wing_1']
# surface_shapes = [(num_nodes, nx, ny, 3), (num_nodes, nx, ny, 3)]

surface_names = ['wing']
surface_shapes = [(num_nodes, nx, ny, 3)]

chord = 2
span = 30 / chord
# https://github.com/LSDOlab/nasa_uli_tc1/blob/222d877228b609076dd352945f4cfe2d158d4973/execution_scripts/c172_climb.py#L33

mesh_dict = {
    "num_y": ny,
    "num_x": nx,
    "wing_type": "rect",
    "symmetry": False,
    "span": span,
    "root_chord": chord,
    "span_cos_spacing": False,
    "chord_cos_spacing": False,
}

# Generate mesh of a rectangular wing
mesh = generate_mesh(mesh_dict) #(nx,ny,3)
offset = span / 2

# mesh_val = generate_simple_mesh(nx, ny, num_nodes)
mesh_val = np.zeros((num_nodes, nx, ny, 3))
# mesh_val_1 = np.zeros((num_nodes, nx, ny, 3))

for i in range(num_nodes):
    mesh_val[i, :, :, :] = mesh
    mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0]
    mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1] + offset

    # mesh_val_1[i, :, :, :] = mesh.copy()
    # mesh_val_1[i, :, :, 1] = mesh.copy()[:, :, 1] - offset
    # mesh_val_1[i, :, :, 0] = mesh.copy()[:, :, 0]

wing = model_1.create_input('wing', val=mesh_val)
# wing_1 = model_1.create_input('wing_1', val=mesh_val_1)
# '''

####################################################################
# 2. preprocessing to connect to the vlm solver
####################################################################

rot_vel = model_1.create_input(surface_names[0] + '_rot_vel',
                               val=np.zeros((num_nodes, nx, ny, 3)))

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

# rho = model_1.create_input('rho', val=0.96 * np.ones((num_nodes, 1)))

eval_pts_shapes = [(num_nodes, x[1] - 1, x[2] - 1, 3) for x in surface_shapes]
submodel = VLMSolverModel(
    surface_names=surface_names,
    surface_shapes=surface_shapes,
    num_nodes=num_nodes,
    # free_stream_velocities=free_stream_velocities,
    eval_pts_location=0.25,  
    coeffs_aoa = [(0.467, 6.37)],
    coeffs_cd = [(7.76E-03 , 0.0189, 0.377)],    
    eval_pts_shapes=eval_pts_shapes,
    AcStates=AcStates_vlm,
    cl0=[0.]
)

# 6.37*x + 0.467
# 7.76E-03 + 0.0189x + 0.377x^2
model_1.add(submodel, 'VLMSolverModel')


# rep = GraphRepresentation(model_1)
# rep.visualize_graph()
# rep.visualize_adjacency_mtx(markersize=0.1)
# rep.visualize_unflat_graph()

sim = Simulator(model_1)
# sim = csdl_lite.Simulator(model_1)

sim.run()



print(
    '=========================\n VLM outputs\n========================='
)



print('#'*100)
print('print outputs\n')
print('F',sim['F'])
print('wing_L',sim['wing_L'])
# print('wing_1_L',sim['wing_1_L'])
print('wing_D',sim['wing_D'])
print('wing_C_L',sim['wing_C_L'])
print('wing_C_D_i',sim['wing_C_D_i'])
print('F',sim['F'])
print('total_drag',sim['total_drag'])
print('total_lift',sim['total_lift'])
print('total_CD',sim['total_CD'])
print('total_CL',sim['total_CL'])
# print('wing_1_D',sim['wing_1_D'])
import pyvista as pv
############################################
# Plot the lifting surfaces
############################################
pv.global_theme.axes.show = True
pv.global_theme.font.label_size = 1
x = mesh[:, :, 0]
y = mesh[:, :, 1]
z = mesh[:, :, 2]
# x_1 = wing_2_mesh[0, :, :, 0]
# y_1 = wing_2_mesh[0, :, :, 1]
# z_1 = wing_2_mesh[0, :, :, 2]

# xw = sim['wing_1_wake_coords'][0, :, :, 0]
# yw = sim['wing_1_wake_coords'][0, :, :, 1]
# zw = sim['wing_1_wake_coords'][0, :, :, 2]

# xw_1 = sim['wing_2_wake_coords'][0, :, :, 0]
# yw_1 = sim['wing_2_wake_coords'][0, :, :, 1]
# zw_1 = sim['wing_2_wake_coords'][0, :, :, 2]

grid = pv.StructuredGrid(x, y, z)
# grid_1 = pv.StructuredGrid(x_1, y_1, z_1)
# gridw = pv.StructuredGrid(xw, yw, zw)
# gridw_1 = pv.StructuredGrid(xw_1, yw_1, zw_1)
p = pv.Plotter()
p.add_mesh(grid, color="blue", show_edges=True, opacity=.5)
# p.add_mesh(gridw, color="blue", show_edges=True, opacity=.5)
# p.add_mesh(grid_1, color="red", show_edges=True, opacity=.5)
# p.add_mesh(gridw_1, color="red", show_edges=True, opacity=.5)
p.camera.view_angle = 60.0
p.add_axes_at_origin(labels_off=True, line_width=5)

# p.show()
