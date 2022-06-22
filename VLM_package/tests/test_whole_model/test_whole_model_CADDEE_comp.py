from functools import partial
import unittest
from VLM_package.VLM_preprocessing.mesh_preprocessing_comp import MeshPreprocessingComp

from VLM_package.VLM_system.vlm_system import VLMSystem
from VLM_package.VLM_outputs.compute_force.compute_outputs_group import Outputs

from csdl import Model
from VLM_package.VLM_preprocessing.utils.generate_simple_mesh import generate_simple_mesh
from VLM_package.examples.run_vlm.AcStates_enum_vlm import *

import csdl
import csdl_om
import numpy as np


class TestVLMModel(unittest.TestCase):
    def initialization(self):
        self.nx, self.ny = 2, 3
        self.nx_1, self.ny_1 = 2, 4
        self.num_nodes = 2
        self.surface_names = ['wing_0', 'wing_1']
        self.surface_shapes = [(self.num_nodes, self.nx, self.ny, 3),
                               (self.num_nodes, self.nx_1, self.ny_1, 3)]
        self.AcStates = AcStates_vlm

    def make_model_add_inputs(self):
        TestVLMModel.initialization(self)

        self.model_1 = Model()

        wing_1_mesh = generate_simple_mesh(self.nx, self.ny, self.num_nodes)

        wing_2_mesh = generate_simple_mesh(self.nx_1,
                                           self.ny_1,
                                           self.num_nodes,
                                           offset=5)

        wing_1_inputs = self.model_1.create_input('wing_0', val=wing_1_mesh)
        wing_2_inputs = self.model_1.create_input('wing_1', val=wing_2_mesh)
        wing_2_inputs = self.model_1.create_input('wing_0_rot_vel',
                                                  val=np.zeros(
                                                      (num_nodes, self.nx,
                                                       self.ny, 3)))
        wing_2_inputs = self.model_1.create_input('wing_1_rot_vel',
                                                  val=np.zeros(
                                                      (num_nodes, self.nx_1,
                                                       self.ny_1, 3)))
        wing_2_inputs = self.model_1.create_input('rho',
                                                  val=np.ones(
                                                      (num_nodes, 1)) * 0.96)

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


class TestVLMModelWhole(TestVLMModel):
    def test_vlm_model_whole(self):
        print()
        print()
        print()

        print(
            '==============================================================================================='
        )
        print('Start TestVLMModelWhole')
        print('---------------------------------------------')
        TestVLMModelWhole.initialization(self)
        TestVLMModelWhole.make_model_add_inputs(self)

        self.model_1.add(
            VLMSystem(
                surface_names=self.surface_names,
                surface_shapes=self.surface_shapes,
                num_nodes=self.num_nodes,
                AcStates=self.AcStates,
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
            AcStates=self.AcStates,
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

        partials = sim.check_partials(out_stream=None)
        sim.prob.check_config(checks=['unconnected_inputs'], out_file=None)

        sim.assert_check_partials(partials, atol=5e-1, rtol=5.e-3)
        del self.model_1
        print('---------------------------------------------')
        print('Finish TestVLMModelWhole')
        print(
            '==============================================================================================='
        )
