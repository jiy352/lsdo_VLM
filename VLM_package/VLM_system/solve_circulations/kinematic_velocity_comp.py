from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size


class KinematicVelocityComp(Model):
    """
    Compute various geometric properties for VLM analysis.
    These are used primarily to help compute postprocessing quantities,
    such as wave CD, viscous CD, etc.
    Some of the quantities, like `normals`, are used to compute the RHS
    of the AIC linear system.

    parameters
    ----------
    
    frame_vel
    p
    q
    r

    Returns
    -------
    ang_vel[num_nodes,3]: p, q, r
    kinematic_vel[n_wake_pts_chord, (num_evel_pts_x* num_evel_pts_y), 3] : csdl array
        undisturbed fluid velocity
    """
    def initialize(self):

        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)

    def define(self):
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        # get num_nodes from surface shape
        num_nodes = surface_shapes[0][0]

        # add_input name and shapes
        # compute rotatonal_vel_shapes from surface shape

        # print('line 49 bd_coll_pts_shapes', bd_coll_pts_shapes)

        # add_output name and shapes
        kinematic_vel_names = [
            x + '_kinematic_vel' for x in self.parameters['surface_names']
        ]

        frame_vel = self.declare_variable('frame_vel', shape=(num_nodes, 3))

        p = self.declare_variable('p_active_nodes', shape=(num_nodes, 1))
        q = self.declare_variable('q_active_nodes', shape=(num_nodes, 1))
        r = self.declare_variable('r_active_nodes', shape=(num_nodes, 1))

        ang_vel = self.create_output('ang_vel', shape=(num_nodes, 3))
        ang_vel[:, 0] = p
        ang_vel[:, 1] = q
        ang_vel[:, 2] = r

        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            num_pts_chord = surface_shapes[i][1]
            num_pts_span = surface_shapes[i][2]
            kinematic_vel_name = kinematic_vel_names[i]
            out_shape = (num_nodes, (num_pts_chord - 1) * (num_pts_span - 1),
                         3)

            coll_pts_coords_name = surface_name + '_coll_pts_coords'

            coll_pts = self.declare_variable(coll_pts_coords_name,
                                             shape=(num_nodes,
                                                    num_pts_chord - 1,
                                                    num_pts_span - 1, 3))
            r_vec = coll_pts - 0.
            ang_vel_exp = csdl.expand(
                ang_vel, (num_nodes, num_pts_chord - 1, num_pts_span - 1, 3),
                indices='il->ijkl')
            rot_vel = csdl.reshape(csdl.cross(ang_vel_exp, r_vec, axis=3),
                                   out_shape)
            frame_vel_expand = csdl.expand(frame_vel,
                                           out_shape,
                                           indices='ij->ikj')
            print('rot_vel shape', rot_vel.shape)
            print('frame_vel_expand shape', frame_vel_expand.shape)

            kinematic_vel = -(rot_vel + frame_vel_expand)
            self.register_output(kinematic_vel_name, kinematic_vel)


if __name__ == "__main__":

    import csdl_lite
    simulator_name = 'csdl_om'
    # simulator_name = 'csdl_lite'

    n_wake_pts_chord = 2
    num_pts_chord = 3
    num_pts_span = 4

    from VLM_package.VLM_preprocessing.mesh_preprocessing_comp import MeshPreprocessingComp
    from VLM_package.VLM_preprocessing.adapter_comp import AdapterComp
    from VLM_package.examples.run_vlm.AcStates_enum_vlm import *

    # add the upstream mesh preprocessing comp
    def generate_simple_mesh(n_wake_pts_chord, num_pts_chord, num_pts_span):
        mesh = np.zeros((n_wake_pts_chord, num_pts_chord, num_pts_span, 3))
        for i in range(n_wake_pts_chord):
            mesh[i, :, :, 0] = np.outer(np.arange(num_pts_chord),
                                        np.ones(num_pts_span))
            mesh[i, :, :, 1] = np.outer(np.arange(num_pts_span),
                                        np.ones(num_pts_chord)).T
            mesh[i, :, :, 2] = 0.
        return mesh

    surface_names = ['wing_1', 'wing_2']

    kinematic_vel_names = [x + '_kinematic_vel' for x in surface_names]
    surface_shapes = [(n_wake_pts_chord, num_pts_chord, num_pts_span, 3),
                      (n_wake_pts_chord, num_pts_chord + 1, num_pts_span + 1,
                       3)]
    model_1 = Model()

    wing_1_mesh = generate_simple_mesh(n_wake_pts_chord, num_pts_chord,
                                       num_pts_span)
    wing_2_mesh = generate_simple_mesh(n_wake_pts_chord, num_pts_chord + 1,
                                       num_pts_span + 1)

    wing_1_inputs = model_1.create_input('wing_1', val=wing_1_mesh)
    wing_2_inputs = model_1.create_input('wing_2', val=wing_2_mesh)

    create_opt = 'create_inputs'

    #creating inputs that share the same names within CADDEE
    for data in AcStates_vlm:
        print('{:15} = {}'.format(data.name, data.value))
        name = data.name
        string_name = data.value
        if create_opt == 'create_inputs':
            variable = model_1.create_input(string_name,
                                            val=AcStates_val_dict[string_name])
            # del variable
        else:
            variable = model_1.declare_variable(
                string_name, val=AcStates_val_dict[string_name])

    model_1.add(MeshPreprocessingComp(surface_names=surface_names,
                                      surface_shapes=surface_shapes),
                name='MeshPreprocessingComp')
    # add the current comp
    model_1.add(AdapterComp(surface_names=surface_names,
                            surface_shapes=surface_shapes),
                name='AdapterComp')
    model_1.add(KinematicVelocityComp(surface_names=surface_names,
                                      surface_shapes=surface_shapes),
                name='KinematicVelocityComp')
    if simulator_name == 'csdl_om':

        sim = Simulator(model_1)

        sim.run()
        # sim.prob.check_partials(compact_print=True)
        partials = sim.prob.check_partials(compact_print=True, out_stream=None)
        sim.assert_check_partials(partials, 1e-5, 1e-7)
        sim.visualize_implementation()
        sim.prob.check_config(checks=['unconnected_inputs'], out_file=None)

    elif simulator_name == 'csdl_lite':
        sim = csdl_lite.Simulator(model_1)

        sim.run()
        sim.check_partials(compact_print=True)

    print('u,v,w,p,q,r\n', sim['u'], sim['v'], sim['w'], sim['p'], sim['q'],
          sim['r'])
    print('frame_vel,alpha,v_inf_sq,beta,kinematic_vel(wing 1, wing2),rho\n',
          sim['frame_vel'], sim['alpha'], sim['v_inf_sq'], sim['beta'],
          sim[kinematic_vel_names[0]], sim[kinematic_vel_names[1]], sim['rho'])
