import csdl
from csdl_om import Simulator
from ozone2.api import NativeSystem
import numpy as np
# from solve_group_simple_wing import SolveMatrix
from UVLM_package.UVLM_system.solve_circulations.solve_group_new import SolveMatrix
from UVLM_package.UVLM_preprocessing.mesh_preprocessing_comp import MeshPreprocessing

"""
This script contains 3 possible ways on defining the same ODE function dydt = f(y) to use for the integrator
1. CSDL model
2. NativeSystem with dense partials
3. NativeSystem with sparse partials

We can easily swap out these three different methods by setting
self.ode_system = 'ode system model' in the ODEProblem class
"""


# ------------------------- METHOD 1: CSDL -------------------------
# very easy to write. No need to write analytical derivatives but potentially worse performance than Native System
class ODESystemModel(csdl.Model):
    def initialize(self):
        # Required every time for ODE systems or Profile Output systems
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('delta_t', default=1)

        # We also passed in parameters to this ODE model in ODEproblem.create_model() in 'run.py' which we can access here.
        # for now, we just make frame_vel, bd_vortex_coords, as static parameters
        self.parameters.declare('bd_vortex_coords', types=list)
        self.parameters.declare('frame_vel')
        self.parameters.declare('wake_coords', types=list)
        self.parameters.declare('nt', types=int)

    def define(self):
        # Input: state
        n = self.parameters['num_nodes']
        nt = self.parameters['nt']
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']


        bd_vortex_shapes = surface_shapes
        delta_t = self.parameters['delta_t']

        bd_vortex_coords_val = self.parameters['bd_vortex_coords']
        frame_vel_val = self.parameters['frame_vel']
        wake_coords_val = self.parameters['wake_coords']

        frame_vel = self.create_input('frame_vel', val=frame_vel_val)
        ode_surface_shape = [(n,) + item for item in surface_shapes]

        ode_bd_vortex_shapes = ode_surface_shape

        # wake_coords = model_1.create_input('wake_coords', val=wake_coords_val)
        for i in range(len(bd_vortex_coords_val)):
            nx = bd_vortex_shapes[i][0]
            ny = bd_vortex_shapes[i][1]
            print('nx, ny', bd_vortex_shapes[i],nx,ny)

            self.create_input(surface_names[i], shape=ode_surface_shape[i])

            bd_vortex_coord_val = bd_vortex_coords_val[i]
            coll_pts_val = 0.25 * (bd_vortex_coord_val[0:nx-1, 0:ny-1, :] +\
                                                    bd_vortex_coord_val[0:nx-1, 1:ny, :] +\
                                                    bd_vortex_coord_val[1:, 0:ny-1, :]+\
                                                    bd_vortex_coord_val[1:, 1:, :])
            print('bd_vortex_coord_valshape------------',
                  bd_vortex_coord_val.shape)
            # model = csdl.Model()
            # bd_vortex_coords = model.create_input('bd_vortex_coords',
            #                                       val=bd_vortex_coords_val)
            # wake_coords = model.create_input('wake_coords',
            #                                  val=wake_coords_val)
            # coll_coords = model.create_input('coll_coords', val=coll_pts_val)
            # self.add(model, 'inputs')
        
            
        self.add(MeshPreprocessing(surface_names=surface_names, surface_shapes=ode_surface_shape), 'meshPreprocessing_comp')
        
        self.add(SolveMatrix(nt=nt,
                                bd_vortex_shapes=bd_vortex_shapes), name='solve_gamma_b_group')

        delta_t = self.parameters['delta_t']
        gamma_b = self.declare_variable('gamma_b',
                                        shape=((nx - 1) * (ny - 1), ))
        # gamma_w = self.declare_variable('gamma_w',
        # shape=(n, nt - 1, ny - 1))
        # gamma_w = self.create_input('gamma_w',
        #                             val=np.zeros((n, nt - 1, ny - 1)))

        val = np.zeros((n, nt - 1, ny - 1))
        gamma_w = self.create_input('gamma_w', val=val)
        dgammaw_dt = self.create_output('dgammaw_dt',
                                        shape=(n, nt - 1, ny - 1))

        for i in range(n):
            gamma_b_last = csdl.reshape(gamma_b[(nx - 2) * (ny - 1):],
                                        new_shape=(1, 1, ny - 1))

            # print('gamma w', gamma_w.val)
            # TODO: for now just let the last row of gamma_b to be a function of gamma_w
            # outputs['dgammaw_dt'][i][
            #     0, :] = gamma_b[i][-1, :] - gamma_w[i][0, :]
            # outputs['dgammaw_dt'][i][0, :] = gamma_b - gamma_w[i][0, :]
            dgammaw_dt[i, 0, :] = gamma_b_last - gamma_w[i, 0, :]

            dgammaw_dt[i,
                        1:, :] = (gamma_w[i, :(gamma_w.shape[1] - 1), :] -
                                    gamma_w[i, 1:, :]) / delta_t
            # dgammaw_dt[i,
            #            1:, :] = (gamma_w[i, :(gamma_w.shape[0] - 1), :] -
            #                      gamma_w[i, 1:, :]) / delta_t



def generate_simple_mesh(nx, ny, nt=None):
    if nt == None:
        mesh = np.zeros((nx, ny, 3))
        mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny)) + 0.25
        mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
        mesh[:, :, 2] = 0.
    else:
        mesh = np.zeros((nt, nx, ny, 3))
        for i in range(nt):
            mesh[i, :, :, 0] = np.outer(np.arange(nx), np.ones(ny)) + 0.25
            mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
            mesh[i, :, :, 2] = 0.
    return mesh

if __name__ == "__main__":

    def generate_simple_mesh(nx, ny, nt=None):
        if nt == None:
            mesh = np.zeros((nx, ny, 3))
            mesh[:, :, 0] = np.outer(np.arange(nx), np.ones(ny)) + 0.25
            mesh[:, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
            mesh[:, :, 2] = 0.
        else:
            mesh = np.zeros((nt, nx, ny, 3))
            for i in range(nt):
                mesh[i, :, :, 0] = np.outer(np.arange(nx), np.ones(ny)) + 0.25
                mesh[i, :, :, 1] = np.outer(np.arange(ny), np.ones(nx)).T
                mesh[i, :, :, 2] = 0.
        return mesh

    # model_1 = Model(bd_vortex_names='bd_vortex_coords',
    #                 bd_vortex_shapes=[(3, 4, 3)])

    # frame_vel_val = np.array([1, 0, 1])
    bd_vortex_coords_val = generate_simple_mesh(3, 4)
    # bd_vortex_coords_val[:, :, 0] = bd_vortex_coords_val[:, :, 0]
    delta_t = 1
    nt = 5
    nx = 3
    ny = 4
    # wake_coords_val = np.array([
    #     [2., 0., 0.],
    #     [2., 1., 0.],
    #     [2., 2., 0.],
    #     [2., 3., 0.],
    #     [42., 0., 0.],
    #     [42., 1., 0.],
    #     [42., 2., 0.],
    #     [42., 3., 0.],
    # ]).reshape(2, 4, 3)

    frame_vel_val = np.array([-1, 0, -1])
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
        generate_simple_mesh(3, 4)[-1, :, 1],
        np.ones(nt),
    ) + (delta_t * np.arange(nt) *
         (-frame_vel_val[1])).reshape(-1, 1)).flatten()
    # bd_vortex_coords_val = generate_simple_mesh(3, 4)
    # wake_coords_val = np.array([
    #     [2., 0., 0.],
    #     [2., 1., 0.],
    #     [2., 2., 0.],
    #     [2., 3., 0.],
    #     [42., 0., 0.],
    #     [42., 1., 0.],
    #     [42., 2., 0.],
    #     [42., 3., 0.],
    # ]).reshape(2, 4, 3)
    wake_coords_val = np.zeros((nt, ny, 3))
    wake_coords_val[:, :, 0] = wake_coords_val_x.reshape(nt, ny)
    wake_coords_val[:, :, 1] = wake_coords_val_y.reshape(nt, ny)
    wake_coords_val[:, :, 2] = wake_coords_val_z.reshape(nt, ny)

    sim = Simulator(
        ODESystemModel(
            bd_vortex_names=['bd_vortex_coords'],
            bd_vortex_shapes=[(3, 4, 3)],
            bd_vortex_coords=[bd_vortex_coords_val],
            frame_vel=frame_vel_val,
            wake_coords=[wake_coords_val],
        ))
    sim.run()
    print(sim['gamma_b'], 'gamma_b')
    print(sim['b'], 'b')
    print(sim['M'], sim['M'].shape, 'M')
    print(sim['gamma_w'], 'gamma_w')
    sim['aic_bd_proj'] @ sim['gamma_b'] + sim['M'] @ sim['gamma_w'].flatten(
    ) + sim['b']
    # sim.visualize_implementation()
'''
        y = self.create_input('y', shape=n)
        x = self.create_input('x', shape=n)

        # Predator Prey ODE:
        a = self.parameters['a']
        b = 0.5
        g = 2.0
        d = 0.5
        dy_dt = a * y - b * y * x
        dx_dt = g * x * y - d * x

        # Register output
        self.register_output('dy_dt', dy_dt)
        self.register_output('dx_dt', dx_dt)


# ------------------------- METHOD 2: NATIVESYSTEM -------------------------
# ODE Model with Native System:
# Need to define partials unlike csdl but better performance


class ODESystemNative(NativeSystem):
    # Setup sets up variables. similar to ExplicitComponnent in OpenMDAO
    def setup(self):
        # NativeSystem does not require an initialization to access parameters
        n = self.num_nodes

        # Need to have ODE shapes similar as first example
        self.add_input('y', shape=n)
        self.add_input('x', shape=n)
        self.add_output('dy_dt', shape=n)
        self.add_output('dx_dt', shape=n)

    # compute the ODE function. similar to ExplicitComponnent in OpenMDAO
    def compute(self, inputs, outputs):
        n = self.num_nodes

        # We have accessed a parameter passed in through the ODEproblem
        a = self.parameters['a']
        b = 0.5
        g = 2.0
        d = 0.5

        # Outputs
        outputs['dy_dt'] = a * inputs['y'] - b * np.multiply(
            inputs['y'], inputs['x'])
        outputs['dx_dt'] = g * np.multiply(inputs['y'],
                                           inputs['x']) - d * inputs['x']

    def compute_partials(self, inputs, partials):
        n = self.num_nodes
        a = self.parameters['a']
        b = 0.5
        g = 2.0
        d = 0.5

        # The partials to compute.
        partials['dy_dt']['y'] = np.diag(a - b * inputs['x'])
        partials['dy_dt']['x'] = np.diag(-b * inputs['y'])
        partials['dx_dt']['y'] = np.diag(g * inputs['x'])
        partials['dx_dt']['x'] = np.diag(g * inputs['y'] - d)

        # The structure of partials has the following for n = self/num_nodes =  4:
        # d(dy_dt)/dy =
        # [d(dy_dt1)/dy1  0               0               0            ]
        # [0              d(dy_dt2)/dy2   0               0            ]
        # [0              0               d(dy_dt2)/dy2   0            ]
        # [0              0               0               d(dy_dt2)/dy2]
        # Hence the diagonal


# ------------------------- METHOD 3: NATIVESYSTEM -------------------------
# ODE Models with Native System allows users to customize types of partials derivatives:
# Partial derivative properties can be set in the setup method


class ODESystemNativeSparse(NativeSystem):
    def setup(self):
        n = self.num_nodes
        self.add_input('y', shape=n)
        self.add_input('x', shape=n)
        self.add_output('dy_dt', shape=n)
        self.add_output('dx_dt', shape=n)

        # Here we define our partial derivatives to be sparse with fixed indices for rows and columns that we define here
        rows = np.arange(n)
        cols = np.arange(n)
        self.declare_partial_properties('dy_dt', 'y', rows=rows, cols=cols)
        self.declare_partial_properties('dy_dt', 'x', rows=rows, cols=cols)
        self.declare_partial_properties('dx_dt', 'y', rows=rows, cols=cols)
        self.declare_partial_properties('dx_dt', 'x', rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        # The compute method is the same as METHOD 2
        n = self.num_nodes
        a = self.parameters['a']
        b = 0.5
        g = 2.0
        d = 0.5

        outputs['dy_dt'] = a*inputs['y'] - b * \
            np.multiply(inputs['y'], inputs['x'])
        outputs['dx_dt'] = g * \
            np.multiply(inputs['y'], inputs['x']) - d*inputs['x']

    def compute_partials(self, inputs, partials):
        n = self.num_nodes
        a = self.parameters['a']
        b = 0.5
        g = 2.0
        d = 0.5

        # Here, we define the values of the sparse partial derivative structure defined in set up.
        partials['dy_dt']['y'] = a - b * inputs['x']
        partials['dy_dt']['x'] = -b * inputs['y']
        partials['dx_dt']['y'] = g * inputs['x']
        partials['dx_dt']['x'] = g * inputs['y'] - d

        # In this case, d(dy_dt)/d(y) = spcipy.sparse.csc_matrix((a - b*inputs['x'], (rows, cols)))
'''