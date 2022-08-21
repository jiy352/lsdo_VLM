# import csdl_om
import csdl
import numpy as np


class EinsumLijkLjLik(csdl.CustomExplicitOperation):
    def initialize(self):
        """
        Declare parameters.
        """
        self.parameters.declare("in_name_1")
        self.parameters.declare("in_name_2")
        self.parameters.declare("in_shape")
        self.parameters.declare("out_name")
        # print('finish initialize---------------')

    def define(self):

        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        in_shape = self.parameters["in_shape"]

        out_name = self.parameters["out_name"]

        self.add_output(out_name,
                        shape=(in_shape[0], in_shape[1], in_shape[3]))
        self.add_input(in_name_1, shape=in_shape)
        self.add_input(in_name_2, shape=(in_shape[0], in_shape[2]))

        rows_1 = np.einsum(
            'ilk,j->iljk',
            np.arange(in_shape[0] * in_shape[1] * in_shape[3]).reshape(
                in_shape[0], in_shape[1], in_shape[3]),
            np.ones(in_shape[2])).flatten()

        cols_1 = np.arange(in_shape[0] * in_shape[1] * in_shape[2] *
                           in_shape[3])

        rows_2 = np.einsum(
            'ik,j->ijk',
            np.arange(in_shape[0] * in_shape[1] * in_shape[3]).reshape(
                in_shape[0], in_shape[1] * in_shape[3]),
            np.ones(in_shape[2])).flatten()
        cols_2 = np.einsum(
            'ik,j->ikj',
            np.arange(in_shape[0] * in_shape[2]).reshape(
                in_shape[0], in_shape[2]),
            np.ones(in_shape[1] * in_shape[3])).flatten()
        self.declare_derivatives(out_name, in_name_1, rows=rows_1, cols=cols_1)
        self.declare_derivatives(out_name, in_name_2, rows=rows_2, cols=cols_2)
        # print('finish define----------------')

    def compute(self, inputs, outputs):
        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        out_name = self.parameters["out_name"]

        outputs[out_name] = np.einsum(
            'lijk,lj->lik',
            inputs[in_name_1],
            inputs[in_name_2],
        )
        # print(outputs[out_name])

    def compute_derivatives(self, inputs, derivatives):
        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        in_shape = self.parameters["in_shape"]
        out_name = self.parameters["out_name"]

        derivatives[out_name,
                    in_name_1] = np.einsum('lj,ik->lijk', inputs[in_name_2],
                                           np.ones((in_shape[1],
                                                    in_shape[3]))).flatten()
        derivatives[out_name, in_name_2] = np.moveaxis(inputs[in_name_1], 2,
                                                       1).flatten()


# l = 2
# i = 3
# j = 4
# k = 5
# in_shape = (l, i, j, k)
# in_name_1 = 'in_1'
# in_name_2 = 'in_2'
# out_name = 'out'

# in_1_val = np.random.random(in_shape)
# in_2_val = np.random.random((l, j))
# model = csdl.Model()
# a = model.declare_variable(in_name_1, val=in_1_val)
# b = model.declare_variable(in_name_2, val=in_2_val)
# product = csdl.custom(a,
#                       b,
#                       op=EinsumLijkLjLik(in_name_1=in_name_1,
#                                          in_name_2=in_name_2,
#                                          in_shape=in_shape,
#                                          out_name=out_name))

# model.register_output(out_name, product)
# sim = csdl_om.Simulator(model)

# sim.prob.run_model()
# sim.prob.check_totals(of=[out_name],
#                       wrt=[in_name_1, in_name_2],
#                       compact_print=True)
