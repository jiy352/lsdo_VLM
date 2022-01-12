import matplotlib.pyplot as plt
import openmdao.api as om
from ozone2.api import ODEProblem, Wrap, NativeSystem
from UVLM_package.UVLM_system.ode_system import ODESystemModel  #, ODESystemNative, ODESystemNativeSparse
# from ode_system_start import ODESystemModel  #, ODESystemNative, ODESystemNativeSparse
# from ode_outputs_delta_p import Profil.eOutputSystem  #, ODESystemNative, ODESystemNativeSparse
from UVLM_package.UVLM_system.ode_outputs_mls import ProfileOutputSystemModel  #, ODESystemNative, ODESystemNativeSparse
import csdl
import csdl_om
import numpy as np


class ODEProblemTest(ODEProblem):
    def setup(self):

        # Outputs. coefficients for field outputs must be defined as a CSDL variable before the integrator is created in RunModel
        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            surface_shape = surface_shapes[i]
            surface_gamma_w_name = surface_names[i] + '_gamma_w'
            surface_gamma_w_out_name = surface_names[i] + '_gamma_w_out'
            self.add_profile_output(surface_gamma_w_out_name,
                                    state_name=surface_gamma_w_name,
                                    shape=(nt - 1, surface_shape[1] - 1))

        # If dynamic == True, The parameter must have shape = (self.num_times, ... shape of parameter @ every timestep ...)
        # The ODE function will use the parameter value at timestep 't': parameter@ODEfunction[shape_p] = fullparameter[t, shape_p]
        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            surface_shape = surface_shapes[i]
            if dynamic_option == True:
                self.add_parameter(surface_name,
                                   shape=(self.num_times, ) + surface_shape,
                                   dynamic=dynamic_option)
            else:
                self.add_parameter(surface_name,
                                   shape=(1, ) + surface_shape,
                                   dynamic=dynamic_option)

        # Inputs names correspond to respective upstream CSDL variables
        for i in range(len(surface_names)):
            surface_name = surface_names[i]
            surface_shape = surface_shapes[i]
            surface_gamma_w_name = surface_name + '_gamma_w'
            surface_dgamma_w_dt_name = surface_name + '_dgammaw_dt'
            surface_initial_condition_name = surface_name + '_gamam_w_0'
            self.add_state(
                surface_gamma_w_name,
                surface_dgamma_w_dt_name,
                initial_condition_name=surface_initial_condition_name,
                shape=(nt - 1, surface_shape[1] - 1),
            )
        self.add_times(step_vector='h')

        # Define ODE system. We have three possible choices as defined in 'ode_systems.py'. Any of the three methods yield identical results:

        self.ode_system = Wrap(ODESystemModel)  # Uncomment for Method 1
        self.profile_outputs_system = Wrap(ProfileOutputSystemModel)


# The CSDL Model containing the ODE integrator


class RunModel(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('delta_t', default=h_stepsize)

        # We also passed in parameters to this ODE model in ODEproblem.create_model() in 'run.py' which we can access here.
        # for now, we just make frame_vel, bd_vortex_coords, as static parameters
        self.parameters.declare('bd_vortex_coords', types=list)
        self.parameters.declare('frame_vel')
        self.parameters.declare('wake_coords', types=list)

        self.parameters.declare('num_timesteps')

    def define(self):
        num_times = self.parameters['num_timesteps']
        # num_nodes = self.parameters['num_nodes']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        delta_t = self.parameters['delta_t']
        bd_vortex_coords = self.parameters['bd_vortex_coords']
        frame_vel = self.parameters['frame_vel']
        wake_coords = self.parameters['wake_coords']

        num_times = self.parameters['num_timesteps']
        # Create parameter for parameters for input vlm meshes
        for i in range(len(surface_names)):
            surface_shape = surface_shapes[i]
            surface_name = surface_names[i]
            # Add to csdl model which are fed into ODE Model
            if dynamic_option == True:
                self.create_input(surface_name,
                                  val=generate_simple_mesh(
                                      surface_shape[0], surface_shape[1],
                                      self.num_times))
            else:
                self.create_input(surface_name,
                                  val=generate_simple_mesh(
                                      surface_shape[0], surface_shape[1], 1))

        h_stepsize = 1

        # Create given inputs
        # self.create_input('coefficients',
        #                   np.ones(num_times + 1) / (num_times + 1))
        # Initial condition for state
        self.create_input('coefficients',
                          np.ones(num_times + 1) / (num_times + 1))
        for i in range(len(surface_names)):
            surface_initial_condition_name = surface_name + '_gamam_w_0'
            surface_shape = surface_shapes[i]
            self.create_input(surface_initial_condition_name,
                              np.zeros((nt - 1, surface_shape[1] - 1)))

        # Timestep vector
        h_vec = np.ones(num_times) * h_stepsize
        self.create_input('h', h_vec)

        # Create model containing integrator
        # We can also pass through parameters to the ODE system from this model.
        params_dict = {
            'surface_names': surface_names,
            'surface_shapes': surface_shapes,
            'delta_t': delta_t,
            'frame_vel': frame_vel,
            'wake_coords': [wake_coords_val, wake_coords_val],
            'nt': nt,
        }
        profile_params_dict = {
            'nt':
            nt,
            'surface_names':
            surface_names,
            'surface_shapes':
            surface_shapes,
            'frame_vel_val':
            frame_vel,
            # 'delta_t': delta_t,
            'surface_coords':
            [generate_simple_mesh(surface_shape[0], surface_shape[1], 1)],
            'wake_coords': [wake_coords_val, wake_coords_val],
        }

        # ODEProblem_instance
        ODEProblem = ODEProblemTest(
            'ForwardEuler',
            'time-marching',
            num_times,
        )
        # visualization='during')
        self.add(
            ODEProblem.create_model(ODE_parameters=params_dict,
                                    profile_parameters=profile_params_dict),
            'subgroup', ['*'])


def generate_simple_mesh(nx, ny, nt=None):
    if nt == None:
        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny))
        mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
        mesh[:, :, 2] = 0.
    else:
        mesh = np.zeros((nt, nx, ny, 3))
        for i in range(nt):
            mesh[i, :, :, 0] = np.outer(np.arange(nx), np.ones(ny))
            mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
            mesh[i, :, :, 2] = 0.
    return mesh


nt = 10
nx = 3
ny = 20
h_stepsize = 1.
dynamic_option = False
# surface_names = ['wing', 'wing1']
surface_names = ['wing']
# surface_shapes = [(nx, ny, 3), (nx, ny, 3)]
surface_shapes = [(nx, ny, 3)]

frame_vel_val = np.array([-1, 0, -1])
delta_t = h_stepsize

wake_coords_val_x = np.einsum(
    'i,j->ij',
    (2.25 + delta_t * np.arange(nt) * (-frame_vel_val[0])),
    np.ones(ny),
).flatten()

wake_coords_val_z = np.einsum(
    'i,j->ij',
    (0 + delta_t * np.arange(nt) * (-frame_vel_val[2])),
    np.ones(ny),
).flatten()

wake_coords_val_y = (np.einsum(
    'i,j->ji',
    generate_simple_mesh(nx, ny)[-1, :, 1],
    np.ones(nt),
) + (delta_t * np.arange(nt) * (-frame_vel_val[1])).reshape(-1, 1)).flatten()
bd_vortex_coords_val = generate_simple_mesh(nx, ny)

wake_coords_val = np.zeros((nt, ny, 3))
wake_coords_val[:, :, 0] = wake_coords_val_x.reshape(nt, ny)
wake_coords_val[:, :, 1] = wake_coords_val_y.reshape(nt, ny)
wake_coords_val[:, :, 2] = wake_coords_val_z.reshape(nt, ny)

# Simulator Object: Note we are passing in a parameter that can be used in the ode system
sim = csdl_om.Simulator(
    RunModel(
        num_timesteps=nt - 1,
        # num_nodes=1,
        surface_names=surface_names,
        surface_shapes=surface_shapes,
        delta_t=1,
        # bd_vortex_coords=[bd_vortex_coords_val, bd_vortex_coords_val],
        bd_vortex_coords=[bd_vortex_coords_val],
        frame_vel=frame_vel_val,
        # wake_coords=[wake_coords_val, wake_coords_val],
        wake_coords=[wake_coords_val],
    ),
    mode='rev')
# sim.prob.set_val(sim['gamma_w'], np.zeros((nt - 1, ny - 1)))
sim.prob.run_model()
# Checktotals
print('wake circulation strength is')
print(sim.prob['wing_gamma_w_out'])
# print('pressure difference is')
# print(sim.prob['delta_p'])

# plot bd_vortex_coords
plt.plot(bd_vortex_coords_val[:, :, 0], bd_vortex_coords_val[:, :, 1])
plt.plot(bd_vortex_coords_val[:, :, 0].T, bd_vortex_coords_val[:, :, 1].T)
plt.savefig('bd_vortex_coords.png')
# sim.visualize_implementation()