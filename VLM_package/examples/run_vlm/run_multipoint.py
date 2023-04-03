# import csdl_lite
# from VLM_package.examples.run_vlm.AcStates_enum_vlm import AcStates_vlm
from numpy import indices
import numpy as np

from VLM_package.VLM_preprocessing.utils.generate_simple_mesh import *

from VLM_package.vlm_solver import VLMSolverModel

from VLM_package.examples.run_vlm.utils.generate_mesh import generate_mesh
import enum
from csdl import GraphRepresentation
from python_csdl_backend import Simulator
# import pyvista as pv

'''
This example demonstrates the VLM simulation for multipoint simulation 
with multiple VLM solvers in a whole model
'''

####################################################################
# 1. Define VLM inputs that share the common names within CADDEE
####################################################################
num_nodes = 2
num_nodes_climb = 1
num_nodes_cruise = 1

class AcStates_vlm(enum.Enum):
    u = 'u'
    v = 'v'
    w = 'w'
    p = 'p'
    q = 'q'
    r = 'r'
    # phi = 'phi'
    theta = 'theta'
    psi = 'psi'
    # x = 'x'
    # y = 'y'
    z = 'z'
    # phiw = 'phiw'
    gamma = 'gamma'
    psiw = 'psiw'
    # rho = 'rho'


# v_inf = np.array([50])
alpha_deg = np.ones((num_nodes,1))*1.25
alpha = alpha_deg / 180 * np.pi
# vx = -v_inf * np.cos(alpha)
# vz = -v_inf * np.sin(alpha)

vx = 50
vz = 2
# vx**2+vz**2=v_inf

AcStates_val_dict = {
    AcStates_vlm.u.value: np.ones((num_nodes,1))*vx,
    AcStates_vlm.v.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.w.value: np.ones((num_nodes,1))*vz,
    AcStates_vlm.p.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.q.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.r.value: np.zeros((num_nodes, 1)),
    # AcStates_vlm.phi.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.theta.value: alpha,
    AcStates_vlm.psi.value: np.zeros((num_nodes, 1)),
    # AcStates_vlm.x.value: np.zeros((num_nodes, 1)),
    # AcStates_vlm.y.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.z.value: np.ones((num_nodes, 1))*1000,
    # AcStates_vlm.phiw.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.gamma.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.psiw.value: np.zeros((num_nodes, 1)),
    # AcStates_vlm.rho.value: np.ones((num_nodes, 1)) * 0.96,
}

####################################################################
# 2. add VLM meshes
####################################################################
# single lifting surface
nx = 3  # number of points in streamwise direction
ny = 41  # number of points in spanwise direction

surface_names = ['wing']
surface_shapes = [(num_nodes_climb, nx, ny, 3)]

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
mesh_val = np.zeros((num_nodes, nx, ny, 3))
for i in range(num_nodes):
    mesh_val[i, :, :, :] = mesh


class RunMultipointModel(Model):
    def define(self):
        # 1. Define inputs
        # 1.1 define aircraft states inputs from the emum (name), and the dictionary (values)
        for data in AcStates_vlm:
            print('{:15} = {}={}'.format(data.name, data.value,AcStates_val_dict[data.value]))
            name = data.name
            string_name = data.value

            variable_climb  = self.create_input(string_name+'climb',
                                            val=AcStates_val_dict[string_name][0:num_nodes_climb,:])   
            variable_cruise = self.create_input(string_name+'cruise',
                                            val=AcStates_val_dict[string_name][num_nodes_climb:, :])                                              
        # 1.2 define mesh input
        wing = self.create_input('wing', val=mesh_val)

        wing_climb   = wing[0,:,:,:]
        wing_cruise  = wing[1,:,:,:]

        self.register_output('wing_climb', wing_climb)
        self.register_output('wing_cruise', wing_cruise)

        eval_pts_shapes = [(num_nodes_climb, x[1] - 1, x[2] - 1, 3) for x in surface_shapes]
        submodel = VLMSolverModel(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            num_nodes=num_nodes_climb,
            eval_pts_location=0.9, 
            coeffs_aoa = [(0.467, 6.37)],
            coeffs_cd = [(7.76E-03 , 0.0189, 0.377)],    
            eval_pts_shapes=eval_pts_shapes,
            AcStates=AcStates_vlm,
            cl0=[0.53])

        self.add(submodel, 'climb_seg',promotes=[])
        self.add(submodel, 'cruise_seg',promotes=[])


        for data in AcStates_vlm:
            print('{:15} = {}={}'.format(data.name, data.value,AcStates_val_dict[data.value]))
            name = data.name
            string_name = data.value
            self.connect(string_name+'climb', 'climb_seg.'+string_name)
            self.connect(string_name+'cruise', 'cruise_seg.'+string_name)
        # 3. Issue connection to the inputs

        self.connect('wing_climb', 'climb_seg.wing')
        self.connect('wing_cruise', 'cruise_seg.wing')

if __name__ == "__main__":
    sim = Simulator(RunMultipointModel())
    sim.run()