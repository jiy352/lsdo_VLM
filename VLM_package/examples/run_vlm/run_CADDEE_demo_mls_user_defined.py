# import csdl_lite
# from VLM_package.examples.run_vlm.AcStates_enum_vlm import AcStates_vlm
from numpy import indices
import numpy as np

from VLM_package.VLM_preprocessing.utils.generate_simple_mesh import *

from VLM_package.vlm_solver import VLMSolverModel

from VLM_package.examples.run_vlm.utils.generate_mesh import generate_mesh
import enum
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


num_nodes = 1

# v_inf = np.array([50])
alpha_deg = np.array([3.125382526501336])
alpha = alpha_deg / 180 * np.pi
# vx = -v_inf * np.cos(alpha)
# vz = -v_inf * np.sin(alpha)

vx = np.array([50.39014388])
vz = np.array([2.75142193])
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
    AcStates_vlm.z.value: np.zeros((num_nodes, 1)),
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
nx = 3  # number of points in streamwise direction
ny = 11  # number of points in spanwise direction

surface_names = ['wing', 'wing_1']
surface_shapes = [(num_nodes, nx, ny, 3), (num_nodes, nx, ny, 3)]

# surface_names = ['wing']
# surface_shapes = [(num_nodes, nx, ny, 3)]

chord = 1.49352
span = 16.2 / chord
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
mesh_val_1 = np.zeros((num_nodes, nx, ny, 3))


for i in range(num_nodes):
    mesh_val[i, :, :, :] = mesh
    mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0]
    mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1] + offset

wing = model_1.create_input('wing', val=mesh_val)

offset_1 = offset*3

for i in range(num_nodes):
    mesh_val_1[i, :, :, :] = mesh.copy()
    mesh_val_1[i, :, :, 0] = mesh.copy()[:, :, 0]
    mesh_val_1[i, :, :, 1] = mesh.copy()[:, :, 1] + offset_1

wing = model_1.create_input('wing_1', val=mesh_val_1)


####################################################################
# 2. preprocessing to connect to the vlm solver
####################################################################

rot_vel = model_1.create_input(surface_names[0] + '_rot_vel',
                               val=np.zeros((num_nodes, nx, ny, 3)))
rot_vel = model_1.create_input(surface_names[1] + '_rot_vel',
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
# '''test #1 2 surfaces, 3 sets of eval pts'''
# eval_pts_shapes = [(num_nodes, nx-1, ny, 3), (num_nodes, nx, ny-1, 3),(num_nodes, nx-1, ny-1, 3)]
# eval_pts_names = ['ev_1', 'ev_2', 'ev_3']

# eval_pts = model_1.create_input(eval_pts_names[0],
#                                val=np.random.random(eval_pts_shapes[0]))

# eval_pts = model_1.create_input(eval_pts_names[1],
#                                val=np.random.random(eval_pts_shapes[1]))

# eval_pts = model_1.create_input(eval_pts_names[2],
#                                val=np.random.random(eval_pts_shapes[2]))

'''test #1 2 surfaces, 1 set of eval pts'''
eval_pts_shapes = [(num_nodes, nx-1, ny, 3)]
eval_pts_names = ['ev_1']

eval_pts = model_1.create_input(eval_pts_names[0],
                               val=np.random.random(eval_pts_shapes[0]))

# eval_pts = model_1.create_input(eval_pts_names[1],
#                                val=np.random.random(eval_pts_shapes[1]))

# eval_pts = model_1.create_input(eval_pts_names[2],
#                                val=np.random.random(eval_pts_shapes[2]))

submodel = VLMSolverModel(
    surface_names=surface_names,
    surface_shapes=surface_shapes,
    num_nodes=num_nodes,
    # free_stream_velocities=free_stream_velocities,
    eval_pts_location=0.25,
    # The location of the evaluation point is on the quarter-chord,
    # if this is not provided, it is defaulted to be 0.25.
    eval_pts_option='user_defined',
    eval_pts_names=eval_pts_names,
    eval_pts_shapes=eval_pts_shapes,
    AcStates=AcStates_vlm,
    cl0=[0.]
)
model_1.add(submodel, 'VLMSolverModel')



sim = Simulator(model_1)
# sim = csdl_lite.Simulator(model_1)

sim.run()

####################################################################
# Print VLM outputs
####################################################################


print('#'*100)
print('print outputs\n')
for i in range(len(eval_pts_names)):
    print('_dynamic_pressure',sim[eval_pts_names[i]+'_dynamic_pressure'].shape,'\n',sim[eval_pts_names[i]+'_dynamic_pressure'])
# print('_dynamic_pressure',sim[eval_pts_names[1]+'_dynamic_pressure'].shape,'\n',sim[eval_pts_names[1]+'_dynamic_pressure'])
# print('_dynamic_pressure',sim[eval_pts_names[2]+'_dynamic_pressure'].shape,'\n',sim[eval_pts_names[2]+'_dynamic_pressure'])
# print('F',sim['F'])
# print('wing_L',sim['wing_L'])
# # print('wing_1_L',sim['wing_1_L'])
# print('wing_D',sim['wing_D'])
# print('wing_C_L',sim['wing_C_L'])
# print('wing_C_D_i',sim['wing_C_D_i'])
# # print('wing_1_D',sim['wing_1_D'])
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

p.show()
