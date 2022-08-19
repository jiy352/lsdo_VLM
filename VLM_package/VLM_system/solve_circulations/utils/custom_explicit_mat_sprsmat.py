import csdl
import numpy as np

from csdl import Model
# from csdl_om import Simulator
from scipy.sparse import coo_array
from scipy import sparse


class M(Model):
    def define(self):
        n = 2
        sol_shape = 3
        num_bd_panel = 6
        num_wake_panel = 3
        ny_last_start = 3

        row = np.arange(num_wake_panel)
        col = (np.arange(num_wake_panel) + ny_last_start)
        data = np.ones(num_wake_panel)
        sprs = coo_array(
            (data, (row, col)),
            shape=(num_wake_panel, num_bd_panel),
        )
        # sprs_all = np.concatenate([sprs.toarray()] * n).reshape(
        #     n, num_wake_panel, num_bd_panel)
        M = self.declare_variable('M_mat',
                                  val=np.random.random(
                                      (n, num_bd_panel, num_wake_panel)))
        M_reshaped = csdl.custom(M,
                                 op=Explicit(
                                     num_nodes=n,
                                     sprs=sprs,
                                     num_bd_panel=num_bd_panel,
                                     num_wake_panel=num_wake_panel,
                                 ))
        self.register_output('M_reshaped', M_reshaped)


# 'ijk,ikl->ijl', M_all(num_modes, num_bd,
#                       num_wake), sprs(num_modes, num_wake, num_bd)


class Explicit(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('num_nodes')  #, types = tuple)
        self.parameters.declare('sprs')  #, types = tuple)
        self.parameters.declare('num_bd_panel')  #, types = tuple)
        self.parameters.declare('num_wake_panel')  #, types = tuple)

    def define(self):
        sprs = self.parameters['sprs']
        num_nodes = self.parameters['num_nodes']
        num_bd_panel = self.parameters['num_bd_panel']
        num_wake_panel = self.parameters['num_wake_panel']

        self.add_input('M_mat',
                       shape=(num_nodes, num_bd_panel, num_wake_panel))

        self.add_output('M_reshaped',
                        shape=(num_nodes, num_bd_panel, num_bd_panel))

        num_row_rep = sprs.shape[0]
        rows = np.outer(
            np.arange(num_nodes * num_bd_panel * num_bd_panel),
            np.ones(num_row_rep),
        )

        num_col_rep = sprs.shape[1]
        cols = np.hstack([
            np.arange(num_nodes * num_bd_panel * num_wake_panel).reshape(
                -1, num_wake_panel)
        ] * num_bd_panel)
        self.declare_derivatives('M_reshaped',
                                 'M_mat',
                                 rows=rows.flatten(),
                                 cols=cols.flatten())

    def compute(self, inputs, outputs):
        sprs = self.parameters['sprs']
        num_nodes = self.parameters['num_nodes']
        num_bd_panel = self.parameters['num_bd_panel']
        num_wake_panel = self.parameters['num_wake_panel']

        outputs['M_reshaped'] = np.einsum('ijk,kl->ijl', inputs['M_mat'],
                                          sprs.todense())

    def compute_derivatives(self, inputs, derivatives):
        sprs = self.parameters['sprs']
        num_nodes = self.parameters['num_nodes']
        num_bd_panel = self.parameters['num_bd_panel']
        num_wake_panel = self.parameters['num_wake_panel']

        derivatives['M_reshaped',
                    'M_mat'] = np.tile(sprs.T.todense().flatten(),
                                       num_nodes * num_bd_panel)

        # sparse.coo_matrix(
        #     (np.tile(sprs.T.todense().flatten(), num_nodes * num_bd_panel)))

        # derivatives['M_reshaped', 'M_mat'] = np.tile(sprs.T.reshape(18, 1),
        #                                          num_nodes * num_bd_panel)


# sim = Simulator(M())
# sim.run()
# M_reshaped = sim['M_reshaped']
# M = sim['M_mat']
# sim.prob.check_partials(compact_print=True)
# # print(np.einsum('ijk,ik->ij', A, x) + b)