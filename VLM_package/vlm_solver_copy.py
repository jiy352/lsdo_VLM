from VLM_package.VLM_system.vlm_system import VLMSystem
from VLM_package.VLM_outputs.compute_force.compute_outputs_group import Outputs
import numpy as np
from lsdo_modules.module_csdl.module_csdl import ModuleCSDL
from caddee.caddee_core.system_model.design_scenario.design_condition.mechanics_group.mechanics_model.mechanics_model import MechanicsModel
from lsdo_modules.module.module import Module
from VLM_package.VLM_preprocessing.utils.generate_simple_mesh import *

# Here n_wake_pts_chord is just a dummy variable that always equal to 2. since we are using a long wake panel,
# we can just make n_wake_pts_chord=2 and delta_t a large number.
class VLM(MechanicsModel):
    def initialize(self, kwargs):
        self.num_nodes = 3
        self.parameters.declare('mesh')
        self.parameters.declare('component')
        self.model_selection = None

    def _assemble_csdl(self):
        mesh = self.parameters['mesh']
        surface_names = mesh.parameters['surface_names']
        surface_shapes = mesh.parameters['surface_shapes']

        component = self.parameters['component']
        prefix = component.parameters['name']

        num_nodes = len(self.model_selection)
        num_active_nodes = int(sum(self.model_selection))
        
        
        surface_shapes = [(num_active_nodes, ) + surface_shape for surface_shape in surface_shapes]
        eval_pts_shapes = [(num_active_nodes, x[1] - 1, x[2] - 1, 3)
                           for x in surface_shapes]

        coeffs_aoa = [(0.535, 0.091), (0.535, 0.091), (0.535, 0.091),
                      (0.535, 0.091)]
        coeffs_cd = [(0.00695, 1.297e-4, 1.466e-4),
                     (0.00695, 1.297e-4, 1.466e-4),
                     (0.00695, 1.297e-4, 1.466e-4),
                     (0.00695, 1.297e-4, 1.466e-4)]
        
        # print(self.num_nodes)
        # exit()
        csdl_model = VLMSolverModel(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            num_nodes=self.num_nodes,
            eval_pts_shapes=eval_pts_shapes,
            # coeffs_aoa=coeffs_aoa,
            AcStates='AcStates',
            # coeffs_cd=coeffs_cd,
            mesh_unit='ft',
            cl0=[0, 0 ,0 ,0],
            model_selection=self.model_selection,
        )

        return csdl_model

class VLMMesh(Module):
    def initialize(self, kwargs):
        self.parameters.declare('surface_names')
        self.parameters.declare('surface_shapes')
        self.parameters.declare('meshes')
        self.parameters.declare('mesh_units', default='m')


class VLMSolverModel(ModuleCSDL):
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('num_nodes', types=int)
        self.parameters.declare('prefix')

        self.parameters.declare('AcStates', default=None)

        self.parameters.declare('free_stream_velocities', default=None)

        self.parameters.declare('eval_pts_location', default=0.25)
        self.parameters.declare('eval_pts_names', default=None)

        self.parameters.declare('eval_pts_option', default='auto')
        self.parameters.declare('eval_pts_shapes', types=list)
        self.parameters.declare('sprs', default=None)
        self.parameters.declare('coeffs_aoa', default=None)
        self.parameters.declare('coeffs_cd', default=None)
        self.parameters.declare('solve_option', default='direct')
        self.parameters.declare('TE_idx', default='last')
        self.parameters.declare('mesh_unit', default='m')
        self.parameters.declare('cl0', default=[0])
        self.parameters.declare('model_selection', types=np.ndarray)

    def define(self):
        # add the mesh info
        surface_names = self.parameters['surface_names']
        num_nodes = self.parameters['num_nodes']
        surface_shapes = self.parameters['surface_shapes']
        model_selection = self.parameters['model_selection']
        num_nodes = len(model_selection)
        num_active_nodes = int(sum(model_selection))
        active_nodes = np.where(model_selection==1)[0]
        # surface_shapes = [(num_active_nodes, ) + surface_shape for surface_shape in surface_shapes]
        
        cl0 = self.parameters['cl0']
        eval_pts_option = self.parameters['eval_pts_option']
        eval_pts_location = self.parameters['eval_pts_location']
        eval_pts_shapes = self.parameters['eval_pts_shapes']
        sprs = self.parameters['sprs']
        coeffs_aoa = self.parameters['coeffs_aoa']
        coeffs_cd = self.parameters['coeffs_cd']
        mesh_unit = self.parameters['mesh_unit']
        

        u_all = self.register_module_input('u', shape=(num_nodes, 1), vectorized=True)
        v_all = self.register_module_input('v', shape=(num_nodes, 1), vectorized=True)
        w_all = self.register_module_input('w', shape=(num_nodes, 1), vectorized=True)
        theta_all = self.register_module_input('theta', shape=(num_nodes, 1), vectorized=True)
        gamma_all = self.register_module_input('gamma', shape=(num_nodes, 1), vectorized=True)
        psi_all = self.register_module_input('psi', shape=(num_nodes, 1), vectorized=True)
        p_all = self.register_module_input('p', shape=(num_nodes, 1), vectorized=True)
        q_all = self.register_module_input('q', shape=(num_nodes, 1), vectorized=True)
        r_all = self.register_module_input('r', shape=(num_nodes, 1), vectorized=True)
        x_all = self.register_module_input('x', shape=(num_nodes, 1), vectorized=True)
        y_all = self.register_module_input('y', shape=(num_nodes, 1), vectorized=True)
        z_all = self.register_module_input('z', shape=(num_nodes, 1), vectorized=True)
        rho_all = self.register_module_input('density', shape=(num_nodes, 1), vectorized=True)

        u = self.register_module_output('u_active_nodes', shape=(num_active_nodes, 1), val=0)
        v = self.register_module_output('v_active_nodes', shape=(num_active_nodes, 1), val=0)
        w = self.register_module_output('w_active_nodes', shape=(num_active_nodes, 1), val=0)
        theta = self.register_module_output('theta_active_nodes', shape=(num_active_nodes, 1), val=0)
        gamma = self.register_module_output('gamma_active_nodes', shape=(num_active_nodes, 1), val=0)
        psi  = self.register_module_output('psi_active_nodes', shape=(num_active_nodes, 1), val=0)
        p = self.register_module_output('p_active_nodes', shape=(num_active_nodes, 1), val=0)
        q = self.register_module_output('q_active_nodes', shape=(num_active_nodes, 1), val=0)
        r = self.register_module_output('r_active_nodes', shape=(num_active_nodes, 1), val=0)
        x = self.register_module_output('x_active_nodes', shape=(num_active_nodes, 1), val=0)
        y = self.register_module_output('y_active_nodes', shape=(num_active_nodes, 1), val=0)
        z = self.register_module_output('z_active_nodes', shape=(num_active_nodes, 1), val=0)
        rho = self.register_module_output('density_active_nodes', shape=(num_active_nodes, 1), val=0)

        for i in range(len(active_nodes)):
            index = int(active_nodes[i])
            u[i, 0] = u_all[index, 0]
            v[i, 0] = v_all[index, 0]
            w[i, 0] = w_all[index, 0]
            theta[i, 0] = theta_all[index, 0]
            gamma[i, 0] = gamma_all[index, 0]
            psi[i, 0] = psi_all[index, 0]
            p[i, 0] = p_all[index, 0]
            q[i, 0] = q_all[index, 0]
            r[i, 0] = r_all[index, 0]
            x[i, 0] = x_all[index, 0]
            y[i, 0] = y_all[index, 0]
            z[i, 0] = z_all[index, 0]
            rho[i, 0] = rho_all[index, 0]


        self.add_module(
            VLMSystem(
                surface_names=surface_names,
                surface_shapes=surface_shapes,
                num_nodes=num_active_nodes,
                AcStates=self.parameters['AcStates'],
                solve_option=self.parameters['solve_option'],
                TE_idx=self.parameters['TE_idx'],
                mesh_unit=mesh_unit,
            ), 'VLM_system')
        if eval_pts_option=='auto':
            eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]
        else:
            eval_pts_names=self.parameters['eval_pts_names']

        # compute lift and drag
        sub = Outputs(
            surface_names=surface_names,
            surface_shapes=surface_shapes,
            eval_pts_names=eval_pts_names,
            eval_pts_shapes=eval_pts_shapes,
            eval_pts_option=eval_pts_option,
            eval_pts_location=eval_pts_location,
            sprs=sprs,
            coeffs_aoa=coeffs_aoa,
            coeffs_cd=coeffs_cd,
            mesh_unit=mesh_unit,
            cl0=cl0,
            num_total_nodes=num_nodes,
            active_nodes=active_nodes,
        )
        self.add(sub, name='VLM_outputs')


if __name__ == "__main__":

    
    pass
