import enum

import numpy as np
import csdl
from modopt.csdl_library import CSDLProblem
from numpy import indices
from python_csdl_backend import Simulator

from VLM_package.VLM_preprocessing.mesh_parameterizartion_model import \
    MeshParameterizationComp
from VLM_package.vlm_solver import VLMSolverModel


'''
This example demonstrates a basic VLM wing shape optimization

Objective: minimize the induced drag coefficient (CDi) of a wing
Design variables: the wing chord (root), taper ratio (λ)), span, pitch angle (θ)
Constraints: the lift coefficient (CL)

'''

####################################################################
# 1. Define VLM inputs
####################################################################

# single lifting surface
num_nodes = 1
nx = 3  # number of points in streamwise direction
ny = 5  # number of points in spanwise direction

surface_names = ['wing']
surface_shapes = [(num_nodes, nx, ny, 3)]

chord = 1.49352
span = 16.2 / chord

# define a model called model_1 for the optimization
model_1 = csdl.Model()

# add inputs to the model
chord_csdl = model_1.create_input('wing_chord_l', val=chord)
span_csdl = model_1.create_input('wing_span_l', val=span)
taper_ratio = model_1.create_input("taper_ratio",val=0.5)

# define constraints as an output to the model
area = (1+taper_ratio)*chord_csdl*span_csdl/2
model_1.register_output('wing_area',area)

# Generate the mesh given the parameters above
submodel = MeshParameterizationComp(
    surface_names=surface_names,
    surface_shapes=surface_shapes,
    )
model_1.add(submodel, 'MeshParameterizationModel')


create_opt = 'create_inputs'
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

alpha_int = np.deg2rad(2)
# define the aircraft states
AcStates_val_dict = {
    AcStates_vlm.u.value: np.ones((num_nodes, 1)),
    AcStates_vlm.v.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.w.value: np.ones((num_nodes, 1)),
    AcStates_vlm.p.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.q.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.r.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.phi.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.theta.value: np.ones((num_nodes, 1))*alpha_int,
    AcStates_vlm.psi.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.x.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.y.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.z.value: np.ones((num_nodes, 1))*1000,
    AcStates_vlm.phiw.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.gamma.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.psiw.value: np.zeros((num_nodes, 1)),
}



for data in AcStates_vlm:
    print('{:15} = {}={}'.format(data.name, data.value,AcStates_val_dict[data.value]))
    name = data.name
    string_name = data.value
    if create_opt == 'create_inputs':
        variable = model_1.create_input(string_name,
                                        val=AcStates_val_dict[string_name])
    else:
        variable = model_1.declare_variable(string_name,
                                            val=AcStates_val_dict[string_name])



####################################################################
# 2. preprocessing to connect to the vlm solver
####################################################################

eval_pts_shapes = [(num_nodes, x[1] - 1, x[2] - 1, 3) for x in surface_shapes]
submodel = VLMSolverModel(
    surface_names=surface_names,
    surface_shapes=surface_shapes,
    num_nodes=num_nodes,
    coeffs_aoa = [(0.467, 6.37)],
    coeffs_cd = [(7.76E-03 , 0.0189, 0.377)],    
    eval_pts_shapes=eval_pts_shapes,
    AcStates=AcStates_vlm,
    cl0=[0.0]
)

model_1.add(submodel, 'VLMSolverModel')

model_1.add_design_variable("wing_chord_l", lower=2, upper=10.0)
model_1.add_design_variable("wing_span_l", lower=3, upper=20.0)
model_1.add_design_variable("theta", lower=np.deg2rad(-10), upper=np.deg2rad(10.0))
model_1.add_design_variable("taper_ratio", lower=0.3, upper=1.0)

model_1.add_constraint("wing_area",equals=20)
model_1.add_constraint("wing_C_L",lower=0.7)

model_1.add_objective("wing_C_D")

sim = Simulator(model_1)

sim.run()

from modopt.scipy_library import SLSQP

# Define problem for the optimization
prob = CSDLProblem(
    problem_name='wing_shape_opt',
    simulator=sim,
)
optimizer = SLSQP(prob, maxiter=20)
optimizer.solve()
# Print results of optimization
optimizer.print_results()

###################################################################
# Print VLM outputs
###################################################################



print(
    '=========================\n print outputs\n========================='
)

print('wing_C_L',sim['wing_C_L'])
print('wing_C_D_i',sim['wing_C_D_i'])
print('total_CL',sim['total_CL'])

print(
    '=========================\n print design variables\n========================='
)
print("wing chord", sim['wing_chord_l'])
print("wing span", sim['wing_span_l'])
print("wing area", sim['wing_area'])

print("theta", np.rad2deg(sim["theta"]))

print("taper_ratio", sim["taper_ratio"])

# visualize the results

import pyvista as pv

############################################
# Plot the lifting surfaces
############################################
pv.global_theme.axes.show = True
pv.global_theme.font.label_size = 1
x = sim[surface_names[0]][:, :,:, 0]
y = sim[surface_names[0]][:,:,:,1]
z = sim[surface_names[0]][:,:,:,2]

grid = pv.StructuredGrid(x, y, z)
p = pv.Plotter()
p.add_mesh(grid, color="blue", show_edges=True, opacity=.5)
p.camera.view_angle = 60.0
p.add_axes_at_origin(labels_off=True, line_width=5)

p.show()
