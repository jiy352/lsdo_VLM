import unittest
from VLM_package.VLM_preprocessing.mesh_preprocessing_comp import MeshPreprocessingComp
from VLM_package.VLM_preprocessing.adapter_comp import AdapterComp
from VLM_package.VLM_preprocessing.wake_coords_comp import WakeCoords
from VLM_package.VLM_system.solve_circulations.kinematic_velocity_comp import KinematicVelocityComp

from csdl import Model
from VLM_package.VLM_preprocessing.utils.generate_simple_mesh import generate_simple_mesh

import csdl
# import csdl_om
import numpy as np
import enum


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


num_nodes = 2

v_inf = np.array([50, 50])
alpha_deg = np.array([2, 4])
alpha = alpha_deg / 180 * np.pi
vx = -v_inf * np.cos(alpha)
vz = -v_inf * np.sin(alpha)

AcStates_val_dict = {
    AcStates_vlm.u.value: vx.reshape(num_nodes, 1),
    AcStates_vlm.v.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.w.value: vz.reshape(num_nodes, 1),
    AcStates_vlm.p.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.q.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.r.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.phi.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.theta.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.psi.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.x.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.y.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.z.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.phiw.value: np.ones((num_nodes, 1)),
    AcStates_vlm.gamma.value: np.ones((num_nodes, 1)),
    AcStates_vlm.psiw.value: np.ones((num_nodes, 1)),
}


class TestVLMModel(unittest.TestCase):
    def initialization(self):
        self.nx, self.ny = 3, 3
        self.nx_1, self.ny_1 = 3, 4
        self.num_nodes = 2
        self.surface_names = ['wing_0', 'wing_1']
        self.surface_shapes = [(self.num_nodes, self.nx, self.ny, 3),
                               (self.num_nodes, self.nx_1, self.ny_1, 3)]

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

        create_opt = 'create_inputs'

        #creating inputs that share the same names within CADDEE
        for data in AcStates_vlm:
            # print('{:15} = {}'.format(data.name, data.value))
            name = data.name
            string_name = data.value
            if create_opt == 'create_inputs':
                variable = self.model_1.create_input(
                    string_name, val=AcStates_val_dict[string_name])
                # del variable
            else:
                variable = self.model_1.declare_variable(
                    string_name, val=AcStates_val_dict[string_name])


class TestMeshPreprocessing(TestVLMModel):
    def test_mesh_prepocessing_comp(self):
        print()
        print()
        print()

        print(
            '==============================================================================================='
        )
        print('Start TestMeshPreprocessing')
        print('---------------------------------------------')
        TestMeshPreprocessing.initialization(self)
        TestMeshPreprocessing.make_model_add_inputs(self)

        self.model_1.add(MeshPreprocessingComp(
            surface_names=self.surface_names,
            surface_shapes=self.surface_shapes),
                         name='preprocessing_group')
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

        sim.assert_check_partials(partials, atol=1.e-5, rtol=1.e-6)
        del self.model_1
        print('---------------------------------------------')
        print('Finish TestMeshPreprocessing')
        print(
            '==============================================================================================='
        )


class TestWakeCoords(TestVLMModel):
    def test_wake_coords_comp(self):
        print()
        print()
        print()

        print(
            '==============================================================================================='
        )
        print('Start TestWakeCoords')
        print('---------------------------------------------')
        TestMeshPreprocessing.initialization(self)
        TestMeshPreprocessing.make_model_add_inputs(self)
        f = self.model_1.create_input('frame_vel',
                                      val=np.array([[-1, 0, -1], [-1, 0, -1]]))

        self.model_1.add(MeshPreprocessingComp(
            surface_names=self.surface_names,
            surface_shapes=self.surface_shapes),
                         name='MeshPreprocessingComp')
        self.model_1.add(WakeCoords(surface_names=self.surface_names,
                                    surface_shapes=self.surface_shapes,
                                    n_wake_pts_chord=2,
                                    delta_t=20),
                         name='WakeCoords')
        sim = csdl_om.Simulator(self.model_1)
        sim.run()
        '''
        '''
        sim.prob.check_config(checks=['unconnected_inputs'], out_file=None)

        partials = sim.check_partials(out_stream=None)

        sim.assert_check_partials(partials, atol=1.e-5, rtol=1.e-6)
        del self.model_1
        print('---------------------------------------------')
        print('Finish TestWakeCoords')
        print(
            '==============================================================================================='
        )


class TestAdapterComp(TestVLMModel):
    def test_adapter_comp(self):
        print()
        print()
        print()

        print(
            '==============================================================================================='
        )
        print('Start TestAdapterComp')
        print('---------------------------------------------')
        TestMeshPreprocessing.initialization(self)
        TestMeshPreprocessing.make_model_add_inputs(self)

        self.model_1.add(MeshPreprocessingComp(
            surface_names=self.surface_names,
            surface_shapes=self.surface_shapes),
                         name='preprocessing_group')
        self.model_1.add(AdapterComp(surface_names=self.surface_names,
                                     surface_shapes=self.surface_shapes),
                         name='AdapterComp')
        sim = csdl_om.Simulator(self.model_1)
        sim.run()
        '''
        '''

        partials = sim.check_partials(out_stream=None)

        sim.assert_check_partials(partials, atol=1.e-5, rtol=1.e-6)
        del self.model_1
        print('---------------------------------------------')
        print('Finish TestAdapterComp')
        print(
            '==============================================================================================='
        )


class TestKinematicVelocityComp(TestVLMModel):
    def test_kinematic_velocith_comp(self):
        print()
        print()
        print()

        print(
            '==============================================================================================='
        )
        print('Start TestKinematicVelocityComp')
        print('---------------------------------------------')
        TestMeshPreprocessing.initialization(self)
        TestMeshPreprocessing.make_model_add_inputs(self)

        self.model_1.add(MeshPreprocessingComp(
            surface_names=self.surface_names,
            surface_shapes=self.surface_shapes),
                         name='preprocessing_group')
        self.model_1.add(AdapterComp(surface_names=self.surface_names,
                                     surface_shapes=self.surface_shapes),
                         name='AdapterComp')
        self.model_1.add(KinematicVelocityComp(
            surface_names=self.surface_names,
            surface_shapes=self.surface_shapes),
                         name='KinematicVelocityComp')

        sim = csdl_om.Simulator(self.model_1)
        sim.run()
        '''
        '''

        partials = sim.check_partials(out_stream=None)

        sim.assert_check_partials(partials, atol=1.e-5, rtol=1.e-6)
        del self.model_1
        print('---------------------------------------------')
        print('Finish TestKinematicVelocityComp')
        print(
            '==============================================================================================='
        )
