import csdl
from csdl_om import Simulator

import numpy as np
from VLM_package.VLM_system.solve_circulations.solve_group import SolveMatrix
from VLM_package.VLM_preprocessing.mesh_preprocessing_comp import MeshPreprocessing
from VLM_package.VLM_preprocessing.compute_wake_coords import WakeCoords

from VLM_package.VLM_system.solve_circulations.seperate_gamma_b import SeperateGammab


class VLMSystemModel(csdl.Model):
    '''
    contains
    1. MeshPreprocessing_comp
    2. WakeCoords_comp
    3. solve_gamma_b_group
    3. seperate_gamma_b_comp
    4. extract_gamma_w_comp
    '''

    def initialize(self):
        self.parameters.declare('num_nodes')
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('delta_t', default=100)

        # We also passed in parameters to this ODE model in ODEproblem.create_model() in 'run.py' which we can access here.
        # for now, we just make frame_vel an option, bd_vortex_coords, as static parameters
        # self.parameters.declare('frame_vel')
        self.parameters.declare('nt', default=2)
        self.parameters.declare('free_wake', default=False)
        self.parameters.declare('temp_fix_option', default=False)
        self.parameters.declare('model_name', default=False)

    def define(self):
        # rename parameters
        num_nodes = self.parameters['num_nodes']
        nt = self.parameters['nt']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        free_wake = self.parameters['free_wake']
        model_name = self.parameters['model_name']

        temp_fix_option = self.parameters['temp_fix_option']

        wake_coords_names = [x + '_wake_coords' for x in surface_names]

        bd_vortex_shapes = surface_shapes
        delta_t = self.parameters['delta_t']
        gamma_b_shape = sum((i[1] - 1) * (i[2] - 1) for i in bd_vortex_shapes)

        # frame_vel = self.declare_variable('frame_vel', shape=(3, ))
        v_total_wake_names = [x + '_wake_total_vel' for x in surface_names]
        wake_vortex_pts_shapes = [
            tuple((nt, item[1], 3)) for item in surface_shapes
        ]
        wake_vel_shapes = [(x[0] * x[1], 3) for x in wake_vortex_pts_shapes]

        self.add(MeshPreprocessing(surface_names=surface_names,
                                   surface_shapes=surface_shapes),
                 name='MeshPreprocessing_comp')
        m = WakeCoords(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            num_nodes=num_nodes,
            nt=nt,
            delta_t=delta_t,
            model_name=model_name,
        )
        # m.optimize_ir(False)
        self.add(m, name='WakeCoords_comp')

        self.add(SolveMatrix(nt=nt,
                             surface_names=surface_names,
                             bd_vortex_shapes=bd_vortex_shapes,
                             delta_t=delta_t,
                             model_name=model_name),
                 name='solve_gamma_b_group')

        gamma_b = self.declare_variable(model_name + 'gamma_b',
                                        shape=(num_nodes, gamma_b_shape))

        self.add(SeperateGammab(surface_names=surface_names,
                                surface_shapes=surface_shapes,
                                model_name=model_name),
                 name='seperate_gamma_b_comp')

        #!TODO:! further check if I can get rid of extract_gamma_w_comp

        # m = csdl.Model()
        # sum_ny = sum((i[1] - 1) for i in bd_vortex_shapes)
        # gamma_w = m.create_output('gamma_w', shape=(num_nodes, nt - 1, sum_ny))
        # start = 0
        # for i in range(len(surface_names)):
        #     nx = bd_vortex_shapes[i][0]
        #     ny = bd_vortex_shapes[i][1]
        #     delta = ny - 1

        #     val = np.zeros((num_nodes, nt - 1, ny - 1))
        #     surface_name = surface_names[i]

        #     surface_gamma_b_name = surface_name + '_gamma_b'

        #     surface_gamma_b = m.declare_variable(surface_gamma_b_name,
        #                                          shape=((nx - 1) * (ny - 1), ))
        #     surface_gamma_w_name = surface_names[i] + '_gamma_w'
        #     surface_gamma_w = csdl.expand(
        #         surface_gamma_b[(nx - 2) * (ny - 1):], (nt - 1, ny - 1),
        #         'i->ji')
        #     m.register_output(surface_gamma_w_name, surface_gamma_w)
        #     gamma_w[:, :, start:start + delta] = csdl.reshape(
        #         surface_gamma_w, (num_nodes, nt - 1, ny - 1))
        #     start += delta
        # self.add(m, name='extract_gamma_w_comp')

        # gamma_b = self.declare_variable('gamma_b', shape=gamma_b_shape)

        # self.add(SeperateGammab(surface_names=surface_names,
        #                         surface_shapes=surface_shapes),
        #          name='seperate_gamma_b')
        # ODE system with surface gamma's
        '''TODO: see if I can delete this'''
        # for i in range(len(surface_names)):
        #     nx = bd_vortex_shapes[i][1]
        #     ny = bd_vortex_shapes[i][2]
        #     val = np.zeros((num_nodes, nt - 1, ny - 1))
        #     surface_name = surface_names[i]

        #     surface_gamma_b_name = surface_name + '_gamma_b'

        #     surface_gamma_b = self.declare_variable(surface_gamma_b_name,
        #                                             shape=(
        #                                                 num_nodes,
        #                                                 (nx - 1) * (ny - 1),
        #                                             ))


if __name__ == "__main__":

    pass
