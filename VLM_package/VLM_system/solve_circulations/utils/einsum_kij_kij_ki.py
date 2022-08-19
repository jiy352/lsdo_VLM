# import csdl_om
import csdl
import numpy as np


class EinsumKijKijKi(csdl.CustomExplicitOperation):
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

        self.add_output(out_name, shape=(in_shape[0], in_shape[1]))
        self.add_input(in_name_1, shape=in_shape)
        self.add_input(in_name_2, shape=in_shape)

        rows = np.outer(np.arange(in_shape[0] * in_shape[1]),
                        np.ones(in_shape[2])).flatten()
        cols = np.arange(in_shape[0] * in_shape[1] * in_shape[2]).flatten()
        # print('rows\n', rows.shape)
        # print('cols\n', cols.shape)
        self.declare_derivatives(out_name, in_name_1, rows=rows, cols=cols)
        self.declare_derivatives(out_name, in_name_2, rows=rows, cols=cols)
        # print('finish define----------------')

    def compute(self, inputs, outputs):
        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        out_name = self.parameters["out_name"]

        outputs[out_name] = np.einsum(
            'kij,kij->ki',
            inputs[in_name_1],
            inputs[in_name_2],
        )
        # print(outputs[out_name])

    def compute_derivatives(self, inputs, derivatives):
        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        out_name = self.parameters["out_name"]

        derivatives[out_name, in_name_1] = inputs[in_name_2].flatten()
        derivatives[out_name, in_name_2] = inputs[in_name_1].flatten()


# k = 3
# i = 2
# j = 4
# in_shape = (k, i, j)
# in_name_1 = 'in_1'
# in_name_2 = 'in_2'
# out_name = 'out'

# in_1_val = np.random.random((k, i, j))
# in_2_val = np.random.random((k, i, j))
# model = csdl.Model()
# a = model.declare_variable(in_name_1, val=in_1_val)
# b = model.declare_variable(in_name_2, val=in_2_val)
# product = csdl.custom(a,
#                       b,
#                       op=EinsumKijKijKi(in_name_1=in_name_1,
#                                         in_name_2=in_name_2,
#                                         in_shape=in_shape,
#                                         out_name=out_name))

# model.register_output(out_name, product)
# sim = csdl_om.Simulator(model)

# sim.prob.run_model()
# sim.prob.check_totals(of=[out_name],
#                       wrt=[in_name_1, in_name_2],
#                       compact_print=True)
