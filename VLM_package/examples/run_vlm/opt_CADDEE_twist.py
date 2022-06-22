import csdl_lite
from VLM_package.examples.run_vlm.AcStates_enum_vlm import AcStates_vlm
from numpy import indices
import numpy as np

from VLM_package.VLM_preprocessing.utils.generate_simple_mesh import *

from VLM_package.vlm_solver import VLMSolverModel

from VLM_package.examples.run_vlm.utils.generate_mesh import generate_mesh

from VLM_package.examples.run_vlm.AcStates_enum_vlm import *

from VLM_package.VLM_preprocessing.geometry.generate_mesh_given_twist import Rotate
from VLM_package.VLM_preprocessing.geometry.generate_spline import BSpline
import openmdao.api as om

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

for data in AcStates_vlm:
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

# surface_names = ['wing', 'wing_1']
# surface_shapes = [(num_nodes, nx, ny, 3), (num_nodes, nx, ny, 3)]

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
mesh = generate_mesh(mesh_dict)
offset = span / 2

# mesh_val = generate_simple_mesh(nx, ny, num_nodes)
mesh_val = np.zeros((num_nodes, nx, ny, 3))
mesh_val_1 = np.zeros((num_nodes, nx, ny, 3))

gamma_b = model_1.create_input('gamma_b',
                               val=np.zeros((num_nodes, (nx - 1) * (ny - 1))))
twist_cp = model_1.create_input('twist_cp', val=np.random.random(3))

model_inputs = csdl.Model()

twist_cp = model_inputs.declare_variable('twist_cp', val=np.random.random(3))
spline = csdl.custom(twist_cp, op=BSpline(ny=ny))
model_inputs.register_output('twist', spline)

in_mesh = model_inputs.create_input('in_mesh', val=mesh)
twist = model_inputs.declare_variable('twist', val=np.random.random(ny))
twisted_mesh = csdl.custom(twist,
                           in_mesh,
                           op=Rotate(val=np.zeros(ny),
                                     mesh_shape=mesh.shape,
                                     symmetry=False))
model_inputs.register_output('mesh', twisted_mesh)

model_1.add(model_inputs, 'inputs_model')
model_1.declare_variable('mesh', shape=(nx, ny, 3))
final_mesh = csdl.expand(twisted_mesh, (num_nodes, nx, ny, 3), 'jkl->ijkl')
model_1.register_output(surface_names[0], final_mesh)

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
    AcStates=AcStates_vlm,
    solve_option='optimization')

model_1.add(submodel, 'VLMSolverModel')

####################################################################
# Print VLM outputs
####################################################################

# for i in range(len(surface_names)):

#     # L_panel_name = surface_names[i] + '_L_panel'
#     # D_panel_name = surface_names[i] + '_D_panel'
#     # L_name = surface_names[i] + '_L'
#     # D_name = surface_names[i] + '_D'
#     # CL_name = surface_names[i] + '_C_L'
#     # CD_name = surface_names[i] + '_C_D_i'
#     # print('lift\n', L_name, sim.prob[L_name])
#     # print('drag\n', D_name, sim.prob[D_name])
#     # # print(
#     # #     'L_panel',
#     # #     L_panel_name,
#     # #     sim.prob[L_panel_name].shape,
#     # #     sim.prob[L_panel_name],
#     # # )
#     # # print(
#     # #     'D_panel',
#     # #     D_panel_name,
#     # #     sim.prob[D_panel_name].shape,
#     # #     sim.prob[D_panel_name],
#     # # )
#     # print('cl\n', CL_name, sim.prob[CL_name])
#     # print('cd\n', CD_name, sim.prob[CD_name])

#     L_panel_name = surface_names[i] + '_L_panel'
#     D_panel_name = surface_names[i] + '_D_panel'
#     L_name = surface_names[i] + '_L'
#     D_name = surface_names[i] + '_D'
#     CL_name = surface_names[i] + '_C_L'
#     CD_name = surface_names[i] + '_C_D_i'
#     print('lift\n', L_name, sim[L_name])
#     print('drag\n', D_name, sim[D_name])
####################################################################
# Visualize n2 diagram (line 188)
####################################################################

# sim.visualize_implementation()
# res = np.einsum('ijk,ik->ij', sim['MTX'], sim['gamma_b']) + sim['b']
# norm = np.linalg.norm(res)
print(
    '=========================\n running check_partials\n========================='
)

# b = sim.check_partials(compact_print=True, out_stream=None)
# sim.assert_check_partials(b, 5e-3, 1e-5)
# c = np.zeros(1220)
# i = 0
# keys = []
# for key in b.keys():
#     c[i] = b[key]['relative_error_norm']
#     keys.append(key)
#     i = i + 1
# sorted_array = np.sort(c)[::-1]
# indices = np.argsort(c)[::-1]
# for i in range(c.size):
#     if (sorted_array[i] > 1e-4) & (sorted_array[i] != np.inf):
#         print(keys[i])
#         print(sorted_array[i])

# {k: v for k, v in sorted(b[b.keys].items(), key=lambda item: item[1])}

# b = sim.check_partials(compact_print=True, out_stream=None)
# sim.assert_check_partials(b, 5e-3, 1e-5)

model_1.declare_variable(surface_names[0] + '_D', shape=(num_nodes, 1))

model_1.add_constraint('residual', equals=0)
model_1.add_constraint('wing_C_L', equals=0.5)

model_1.add_design_variable('gamma_b')
model_1.add_design_variable('twist_cp', upper=100, lower=-100)
model_1.add_objective(surface_names[0] + '_D')

sim = Simulator(model_1)

sim.run()
sim.prob.driver = om.pyOptSparseDriver()

sim.prob.driver.options["optimizer"] = "SNOPT"

driver = sim.prob.driver

driver.options["optimizer"] = "SNOPT"
driver.opt_settings["Verify level"] = 0

driver.opt_settings["Major iterations limit"] = 100
driver.opt_settings["Minor iterations limit"] = 100000
driver.opt_settings["Iterations limit"] = 100000000
driver.opt_settings["Major step limit"] = 2.0

driver.opt_settings["Major feasibility tolerance"] = 1.0e-5
driver.opt_settings["Major optimality tolerance"] = 6.0e-6

sim.prob.run_driver()
sim.prob.check_totals(of='wing_D', wrt='twist_cp', compact_print=True)
sim.prob.check_totals(compact_print=True)
