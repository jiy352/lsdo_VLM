# import csdl_lite
# from VLM_package.examples.run_vlm.AcStates_enum_vlm import AcStates_vlm
from numpy import indices
import numpy as np

from VLM_package.VLM_preprocessing.utils.generate_simple_mesh import *

from VLM_package.vlm_solver_tests import VLMSolverModel

from VLM_package.examples.run_vlm.utils.generate_mesh import generate_mesh
import enum
from python_csdl_backend import Simulator

import time
# import pyvista as pv
'''
This example demonstrates the basic VLM simulation 
with a single lifting surface with internal function to generate evaluation pts
Please see vlm_scipt_mls.py for how to use user defined evaluation pts
'''




import os
import psutil
memory_list = []
memory_list_before = []
# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss
 
# decorator function
def profile(func):
    def wrapper(*args, **kwargs):
 
        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        print("{}:consumed memory: {:,}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before))
        # print('sssssssss',mem_before)
        # print('sssssssss',mem_after)
        # print('sssssssss',mem_after - mem_before)
        # memory_list.append(mem_before)
        memory_list.append(mem_after - mem_before)
        memory_list_before.append(mem_before)
 
        return result
    return wrapper
 
# instantiation of decorator function
@profile
 
# main code for which
# memory has to be monitored
def func(ny):
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
    alpha_deg = np.array([1.25])
    alpha = alpha_deg / 180 * np.pi
    # vx = -v_inf * np.cos(alpha)
    # vz = -v_inf * np.sin(alpha)

    vx = np.array([50.39014388])*1
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
    mesh = generate_mesh(mesh_dict) #(nx,ny,3)
    offset = span / 2

    # mesh_val = generate_simple_mesh(nx, ny, num_nodes)
    mesh_val = np.zeros((num_nodes, nx, ny, 3))
    # mesh_val_1 = np.zeros((num_nodes, nx, ny, 3))

    for i in range(num_nodes):
        mesh_val[i, :, :, :] = mesh
        mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0]
        mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1] + offset


    wing = model_1.create_input('wing', val=mesh_val)


    ####################################################################
    # 2. preprocessing to connect to the vlm solver
    ####################################################################

    rot_vel = model_1.create_input(surface_names[0] + '_rot_vel',
                                val=np.zeros((num_nodes, nx, ny, 3)))


    eval_pts_shapes = [(num_nodes, x[1] - 1, x[2] - 1, 3) for x in surface_shapes]
    submodel = VLMSolverModel(
        surface_names=surface_names,
        surface_shapes=surface_shapes,
        num_nodes=num_nodes,
        # free_stream_velocities=free_stream_velocities,
        eval_pts_location=0.25,
        # The location of the evaluation point is on the quarter-chord,  
        coeffs_aoa = [(0.467, 6.37)],
        coeffs_cd = [(7.76E-03 , 0.0189, 0.377)],    
        eval_pts_shapes=eval_pts_shapes,
        AcStates=AcStates_vlm,
        cl0=[0.53]
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
    del sim
ny = np.arange(5,90,14)
# ny = np.array([ 3, 19, 35, 67])
time_list = []
for i in ny:
    print(i)
    t_s = time.time()
    func(int(i))
    
    time_list.append(time.time() - t_s)
import matplotlib

matplotlib.use('Agg')
from pytikz.matplotlib_utils import use_latex_fonts, get_plt_no_show, save_fig, adjust_spines
plt.tight_layout()

import matplotlib.pyplot as plt
plt.figure(7)
plt.plot(ny, np.array(memory_list)[:],marker='o',)
plt.xlabel('$n_y$',fontsize=20)
plt.ylabel('memory usage (B)',fontsize=20)
ax = plt.gca()
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
plt.tight_layout()
save_fig('memory.pdf')
plt.figure(8)
plt.plot(ny, np.array(time_list)[:],marker='o')
plt.xlabel('$n_y$',fontsize=20)
plt.ylabel('time (s)',fontsize=20)
plt.tight_layout()

ax = plt.gca()
plt.tight_layout()

ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
plt.tight_layout()

save_fig('time.pdf')
# plt.plot(np.array(memory_list)[1:])

# np.savetxt('memory_list.txt',np.array(memory_list)[:])
# np.savetxt('time_list.txt',np.array(time_list)[:])

