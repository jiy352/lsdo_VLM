# from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
# from fluids import atmosphere as atmosphere
from lsdo_atmos.atmosphere_model import AtmosphereModel

class AdapterComp(Model):
    """
    An adapter component that takes in 15 variables from CADDEE (not all are used), 
    and adaptes in to frame_vel(linear without rotation),
    rotational velocity, and air density rho.

    parameters
    ----------
    u[num_nodes,1] : csdl array
        vx of the body
    v[num_nodes,1] : csdl array
        vy of the body
    w[num_nodes,1] : csdl array
        vz of the body

    p[num_nodes,1] : csdl array
        omega_x of the body
    q[num_nodes,1] : csdl array
        omega_y of the body
    r[num_nodes,1] : csdl array
        omega_z of the body

    phi[num_nodes,1] : csdl array
        angular rotations relative to the equilibrium state: p=\dot{phi}
    theta[num_nodes,1] : csdl array
        angular rotations relative to the equilibrium state: q=\dot{theta}
    psi[num_nodes,1] : csdl array
        angular rotations relative to the equilibrium state: r=\dot{psi}

    x[num_nodes,1] : csdl array
        omega_x of the body
    y[num_nodes,1] : csdl array
        omega_y of the body
    z[num_nodes,1] : csdl array
        omega_z of the body

    phiw[num_nodes,1] : csdl array
        omega_x of the body
    gamma[num_nodes,1] : csdl array
        omega_y of the body
    psiw[num_nodes,1] : csdl array
        omega_z of the body    

    collocation points

    Returns
    -------
    1. frame_vel[num_nodes,3] : csdl array
        inertia frame vel
    2. alpha[num_nodes,1] : csdl array
        AOA in rad
    3. v_inf_sq[num_nodes,1] : csdl array
        square of v_inf in rad
    4. beta[num_nodes,1] : csdl array
        sideslip angle in rad
    5. rho[num_nodes,1] : csdl array
        air density

    # s_panel [(num_pts_chord-1)* (num_pts_span-1)]: csdl array
    #     The panel areas.
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)

    def define(self):
        # add_input
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']

        num_nodes = surface_shapes[0][0]

        u = self.declare_variable('u', shape=(num_nodes, 1)) 
        v = self.declare_variable('v', shape=(num_nodes, 1))
        w = self.declare_variable('w', shape=(num_nodes, 1))

        p = self.declare_variable('p', shape=(num_nodes, 1))
        q = self.declare_variable('q', shape=(num_nodes, 1))
        r = self.declare_variable('r', shape=(num_nodes, 1))

        phi = self.declare_variable('phi', shape=(num_nodes, 1))
        theta = self.declare_variable('theta', shape=(num_nodes, 1))
        psi = self.declare_variable('psi', shape=(num_nodes, 1))

        x = self.declare_variable('x', shape=(num_nodes, 1))
        y = self.declare_variable('y', shape=(num_nodes, 1))
        z = self.declare_variable('z', shape=(num_nodes, 1))

        phiw = self.declare_variable('phiw', shape=(num_nodes, 1))
        gamma = self.declare_variable('gamma', shape=(num_nodes, 1))
        psiw = self.declare_variable('psiw', shape=(num_nodes, 1))

        frontAct = self.declare_variable('frontAct', shape=(num_nodes, 1))
        rearAct = self.declare_variable('rearAct', shape=(num_nodes, 1))
        
        ################################################################################
        # Block for checking the inputs into VLM
        ################################################################################

        trashU = u * np.ones((num_nodes, 1))
        trashV = v * np.ones((num_nodes, 1))
        trashW = w * np.ones((num_nodes, 1))

        trashP = p * np.ones((num_nodes, 1))
        trashQ = q * np.ones((num_nodes, 1))
        trashR = r * np.ones((num_nodes, 1))

        trashPhi   = phi * np.ones((num_nodes, 1))
        trashTheta = theta * np.ones((num_nodes, 1))
        trashPsi   = psi * np.ones((num_nodes, 1))
        
        trashX = x * np.ones((num_nodes, 1))
        trashY = y * np.ones((num_nodes, 1))
        trashZ = z * np.ones((num_nodes, 1))

        trashPhiW  = phiw * np.ones((num_nodes, 1))
        trashGamma = gamma * np.ones((num_nodes, 1))
        trashPsiW  = psiw * np.ones((num_nodes, 1))

        tempU = self.register_output('VLM_u', trashU)
        tempV = self.register_output('VLM_v', trashV)
        tempW = self.register_output('VLM_w', trashW)

        tempP = self.register_output('VLM_p', trashP)
        tempQ = self.register_output('VLM_q', trashQ)
        tempR = self.register_output('VLM_r', trashR)

        tempPhi   = self.register_output('VLM_phi', trashPhi)
        tempTheta = self.register_output('VLM_theta', trashTheta)
        tempPsi   = self.register_output('VLM_psi', trashPsi)

        tempX   = self.register_output('VLM_x', trashX)
        tempY   = self.register_output('VLM_y', trashY)
        tempZ   = self.register_output('VLM_z', trashZ)

        tempPhiW   = self.register_output('VLM_phiW', trashPhiW)
        tempGamma  = self.register_output('VLM_gamma', trashGamma)
        tempPsiW   = self.register_output('VLM_psiW', trashPsiW)       

        trashFrontAct = frontAct * np.ones((num_nodes, 1))
        trashRearAct  = rearAct * np.ones((num_nodes, 1))

        tempFrontAct = self.register_output('VLM_frontAct', trashFrontAct)
        tempRearAct = self.register_output('VLM_rearAct', trashRearAct)
        

        # self.print_var(tempU)
        # self.print_var(tempV)
        # self.print_var(tempW)
        # self.print_var(tempP)
        # self.print_var(tempQ)
        # self.print_var(tempR)
        # self.print_var(tempPhi)
        # self.print_var(tempTheta)
        # self.print_var(tempPsi)
        # self.print_var(tempX)
        # self.print_var(tempY)
        # self.print_var(tempZ)
        # self.print_var(tempPhiW)
        # self.print_var(tempGamma)
        # self.print_var(tempPsiW)

        # self.print_var(tempFrontAct)
        # self.print_var(tempRearAct)

        # u_e = u * csdl.cos(theta) + w * csdl.sin(theta)
        # w_e = -u * csdl.sin(theta) + w * csdl.cos(theta)

        # testGamma = csdl.arctan(w_e/u_e) +  np.ones((num_nodes, 1))*1e-12

        testGamma = csdl.arctan(w/u)
        self.register_output('testGamma', testGamma)
        self.print_var(testGamma)
        
        alphaConstraint1 = frontAct - testGamma
        alphaConstraint2 = rearAct - testGamma

        self.register_output('frontConst', alphaConstraint1)
        self.register_output('rearConst', alphaConstraint2)

        ################################################################################
        # compute the output: 3. v_inf_sq (num_nodes,1)
        ################################################################################
        v_inf_sq = (u**2 + v**2 + w**2)
        v_inf = (u**2 + v**2 + w**2)**0.5
        self.register_output('v_inf_sq', v_inf_sq)

        ################################################################################
        # compute the output: 3. alpha (num_nodes,1)
        ################################################################################
        # alpha = frontAct - testGamma
        alpha = theta - gamma
        # alpha = csdl.arctan(w/u)
        # alpha = 
        # self.print_var(theta)
        # self.print_var(gamma)
        self.register_output('alpha', alpha)
        self.print_var(alpha)
        ################################################################################
        # compute the output: 4. beta (num_nodes,1)
        ################################################################################
        beta = psi + psiw
        # we always assume v_inf > 0 here
        tempBeta = self.register_output('beta', beta)
        # self.print_var(tempBeta)
        ################################################################################
        # create the output: 1. frame_vel (num_nodes,3)
        # TODO:fix this
        ################################################################################

        frame_vel = self.create_output('frame_vel', shape=(num_nodes, 3))

        frame_vel[:, 0] = -v_inf * csdl.cos(beta) * csdl.cos(alpha)
        frame_vel[:, 1] = v_inf * csdl.sin(beta)
        frame_vel[:, 2] = -v_inf * csdl.cos(beta) * csdl.sin(alpha)

        self.print_var(frame_vel)

        ################################################################################
        # compute the output: 5. rho
        # TODO: replace this hard coding
        ################################################################################
        # h = 1000
        # atmosisa = atmosphere.ATMOSPHERE_1976(Z=h)
        # rho_val = atmosisa.rho

        self.add(AtmosphereModel(
            shape=(num_nodes,1),
        ),name='atmosphere_model')

        # self.create_input('rho', val=rho_val * np.ones((num_nodes, 1)))

        # self.create_input('rho', val=(num_nodes, 1))


if __name__ == "__main__":

    import csdl_lite
    # simulator_name = 'csdl_om'
    simulator_name = 'csdl_lite'

    n_wake_pts_chord = 2
    num_pts_chord = 3
    num_pts_span = 4

    from VLM_package.VLM_preprocessing.mesh_preprocessing_comp import MeshPreprocessingComp
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
    print('frame_vel,alpha,v_inf_sq,beta,rho\n', sim['frame_vel'],
          sim['alpha'], sim['v_inf_sq'], sim['beta'], sim['rho'])
