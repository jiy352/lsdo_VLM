# import csdl_lite
# from VLM_package.examples.run_vlm.AcStates_enum_vlm import AcStates_vlm
import time
from numpy import indices
import numpy as np

from VLM_package.VLM_preprocessing.utils.generate_simple_mesh import *

from VLM_package.vlm_solver import VLMSolverModel

from VLM_package.examples.run_vlm.utils.generate_mesh import generate_mesh
import enum
from python_csdl_backend import Simulator
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
alpha_deg = np.ones(num_nodes).reshape(-1,1)
alpha = alpha_deg / 180 * np.pi
vx = 50
vz=2


AcStates_val_dict = {
    AcStates_vlm.u.value: np.ones((num_nodes, 1))*vx,
    AcStates_vlm.v.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.w.value: np.ones((num_nodes, 1))*vz,
    AcStates_vlm.p.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.q.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.r.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.phi.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.theta.value: np.ones((num_nodes, 1))*alpha,
    AcStates_vlm.psi.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.x.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.y.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.z.value: np.ones((num_nodes, 1))*1000,
    AcStates_vlm.phiw.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.gamma.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.psiw.value: np.zeros((num_nodes, 1)),
    # AcStates_vlm.rho.value: np.ones((num_nodes, 1)) * 0.96,
}



for data in AcStates_vlm:
    # print('{:15} = {}={}'.format(data.name, data.value,AcStates_val_dict[data.value]))
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
ny = 101  # number of points in spanwise direction


surface_names = ['wing']
surface_shapes = [(num_nodes, nx, ny, 3)]

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

mesh_val = np.zeros((num_nodes, nx, ny, 3))

for i in range(num_nodes):
    mesh_val[i, :, :, :] = mesh
    mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0]
    mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1] + offset


wing = model_1.create_input('wing', val=mesh_val)


eval_pts_shapes = [(num_nodes, x[1] - 1, x[2] - 1, 3) for x in surface_shapes]
submodel = VLMSolverModel(
    surface_names=surface_names,
    surface_shapes=surface_shapes,
    num_nodes=num_nodes,
    # free_stream_velocities=free_stream_velocities,
    eval_pts_location=0.25, 
    coeffs_aoa = None,
    coeffs_cd = None,    
    eval_pts_shapes=eval_pts_shapes,
    AcStates='dummy',
    cl0=[0.]
)

# 6.37*x + 0.467
# 7.76E-03 + 0.0189x + 0.377x^2
model_1.add(submodel, 'VLMSolverModel')




sim = Simulator(model_1)
sim.run()
t_s = time.time()
for i in range(5):
    sim.run()
t_run = time.time()-t_s

print('the run time is:\n', t_run/5)
