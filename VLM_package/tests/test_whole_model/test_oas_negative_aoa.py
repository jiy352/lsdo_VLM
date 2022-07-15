from functools import partial
import unittest
from VLM_package.VLM_preprocessing.mesh_preprocessing_comp import MeshPreprocessingComp

from VLM_package.VLM_system.vlm_system import VLMSystem
from VLM_package.VLM_outputs.compute_force.compute_outputs_group import Outputs

from csdl import Model
from VLM_package.VLM_preprocessing.utils.generate_simple_mesh import generate_simple_mesh
import enum
from VLM_package.examples.run_vlm.utils.generate_mesh import generate_mesh

import csdl
import csdl_om
import numpy as np


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

if num_nodes == 1:
    v_inf = np.array([50])
    alpha_deg = -np.array([4])
    alpha = alpha_deg / 180 * np.pi
    vz = np.array([-50])
    vx = np.array([-1e-6])

    # vx = np.array([50])
    # vz = np.array([1])

    AcStates_val_dict = {
        AcStates_vlm.u.value: vx.reshape(num_nodes, 1),
        AcStates_vlm.v.value: np.zeros((num_nodes, 1)),
        AcStates_vlm.w.value: vz.reshape(num_nodes, 1),
        AcStates_vlm.p.value: np.zeros((num_nodes, 1)),
        AcStates_vlm.q.value: np.zeros((num_nodes, 1)),
        AcStates_vlm.r.value: np.zeros((num_nodes, 1)),
        AcStates_vlm.phi.value: np.zeros((num_nodes, 1)),
        AcStates_vlm.theta.value: np.ones(
            (num_nodes, 1)) * np.deg2rad(alpha_deg),
        AcStates_vlm.psi.value: np.zeros((num_nodes, 1)),
        AcStates_vlm.x.value: np.zeros((num_nodes, 1)),
        AcStates_vlm.y.value: np.zeros((num_nodes, 1)),
        AcStates_vlm.z.value: np.zeros((num_nodes, 1)),
        AcStates_vlm.phiw.value: np.zeros((num_nodes, 1)),
        AcStates_vlm.gamma.value: np.zeros((num_nodes, 1)),
        AcStates_vlm.psiw.value: np.zeros((num_nodes, 1)),
        # AcStates_vlm.rho.value: np.ones((num_nodes, 1)) * 0.96,
    }


class TestVLMModel(unittest.TestCase):
    def initialization(self):
        self.nx, self.ny = 3, 9
        self.num_nodes = 1
        self.surface_names = ['wing']
        self.surface_shapes = [(self.num_nodes, self.nx, self.ny, 3)]
        self.AcStates = AcStates_vlm

    def make_model_add_inputs(self):
        TestVLMModel.initialization(self)

        self.model_1 = Model()

        mesh_dict = {
            "num_y": self.ny,
            "num_x": self.nx,
            "wing_type": "rect",
            "symmetry": False,
            "span": 12,
            "root_chord": 2,
            "span_cos_spacing": False,
            "chord_cos_spacing": False,
        }

        mesh = generate_mesh(mesh_dict)
        mesh_val = mesh

        wing_1_inputs = self.model_1.create_input(self.surface_names[0],
                                                  val=mesh_val.reshape(
                                                      1, self.nx, self.ny, 3))
        wing_2_inputs = self.model_1.create_input('wing_0_rot_vel',
                                                  val=np.zeros(
                                                      (num_nodes, self.nx,
                                                       self.ny, 3)))

        wing_2_inputs = self.model_1.create_input('rho',
                                                  val=np.ones(
                                                      (num_nodes, 1)) * 0.38)

        for data in AcStates_vlm:
            print('{:15} = {}'.format(data.name, data.value))
            name = data.name
            string_name = data.value
            create_opt = 'create_inputs'
            if create_opt == 'create_inputs':
                variable = self.model_1.create_input(
                    string_name, val=AcStates_val_dict[string_name])
                # del variable
            else:
                variable = self.model_1.declare_variable(
                    string_name, val=AcStates_val_dict[string_name])


##############################################################################
# test positive aoa, T.E = mesh[:,-1,:,0]
##############################################################################


class TestVLMModelWholeFirst(TestVLMModel):
    def test_vlm_model_whole(self):
        print()
        print()
        print()

        print(
            '==============================================================================================='
        )
        print('Start TestVLMModelWhole')
        print('---------------------------------------------')
        TestVLMModelWholeFirst.initialization(self)
        TestVLMModelWholeFirst.make_model_add_inputs(self)

        self.model_1.add(
            VLMSystem(
                surface_names=self.surface_names,
                surface_shapes=self.surface_shapes,
                num_nodes=self.num_nodes,
                AcStates='dummy',
            ), 'VLM_system')

        eval_pts_names = [x + '_eval_pts_coords' for x in self.surface_names]
        eval_pts_shapes = [(self.num_nodes, x[1] - 1, x[2] - 1, 3)
                           for x in self.surface_shapes]

        coeffs_aoa = [(0.535, 0.091), (0.535, 0.091)]
        coeffs_cd = [(0.00695, 1.297e-4, 1.466e-4),
                     (0.00695, 1.297e-4, 1.466e-4)]

        # compute lift and drag
        sub = Outputs(
            surface_names=self.surface_names,
            surface_shapes=self.surface_shapes,
            eval_pts_names=eval_pts_names,
            eval_pts_shapes=eval_pts_shapes,
            eval_pts_option='auto',
            eval_pts_location=0.25,
            sprs=None,
            coeffs_aoa=coeffs_aoa,
            coeffs_cd=coeffs_cd,
        )
        self.model_1.add(sub, name='VLM_outputs')

        sim = csdl_om.Simulator(self.model_1)
        sim.run()
        '''

        def_mesh = wing_1_mesh
        bd_vtx_coords = np.zeros(def_mesh.shape)

        bd_vtx_coords[:, 0:nx -
                    1, :, :] = def_mesh[:, 0:nx -
                                        1, :, :] * .75 + def_mesh[:, 1:
                                                                    nx, :, :] * 0.25
        bd_vtx_coords[:, nx - 1, :, :] = def_mesh[:, nx - 1, :, :] + 0.25 * (
            def_mesh[:, nx - 1, :, :] - def_mesh[:, nx - 2, :, :])
        self.assertIsNone(
            np.testing.assert_array_equal(sim['wing_0_bd_vtx_coords'],
                                        bd_vtx_coords))
        '''

        print('The lift is', sim['wing_L'])
        print('The drag is', sim['wing_D'])
        # print('The total drag is', sim['wing_D_total'])
        # print('The lift coeff is', sim['wing_C_L'])
        # print('The induced drag coeff is', sim['wing_C_D_i'])
        # print('The total drag coeff is', sim['wing_C_D'])
        print('The F is', sim['F'])
        print('The frame velocity is', sim['frame_vel'])
        print('The M is', sim['M'])
        F_oas = np.array([-196.39732124, 0.,
                          -3583.875389]).reshape(num_nodes, 3)
        M_oas = np.array([-6.82121026e-13, -1.83978311e+03,
                          5.68434189e-14]).reshape(num_nodes, 3)
        self.assertIsNone(
            np.testing.assert_array_almost_equal(np.linalg.norm(
                (F_oas - sim['F'])) / np.linalg.norm(F_oas),
                                                 0,
                                                 decimal=2))
        self.assertIsNone(
            np.testing.assert_array_almost_equal(np.linalg.norm(
                (M_oas - sim['M'])) / np.linalg.norm(M_oas),
                                                 0,
                                                 decimal=2))
        # partials = sim.check_partials(out_stream=None)
        # sim.assert_check_partials(partials, atol=5e-1, rtol=5.e-3)
        sim.prob.check_config(checks=['unconnected_inputs'], out_file=None)
        del self.model_1
        print('---------------------------------------------')
        print('Finish TestVLMModelWhole')
        print(
            '==============================================================================================='
        )


# Fx = -3765.18404307 * np.sin(np.deg2rad(4)) + 53.26393036 * np.cos(
#     np.deg2rad(4))
# Fx = -3765.18404307 * np.sin(np.deg2rad(4)) + 137.73366131 * np.cos(
#     np.deg2rad(4))
# Fy = 3765.18404307 * np.cos(np.deg2rad(4)) + 53.26393036 * np.sin(
#     np.deg2rad(4))
