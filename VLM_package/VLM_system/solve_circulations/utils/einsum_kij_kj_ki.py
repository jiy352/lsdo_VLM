# import csdl_om
import csdl
import numpy as np


class EinsumKijKjKi(csdl.CustomExplicitOperation):
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
        self.add_input(in_name_2, shape=(in_shape[0], in_shape[2]))

        rows_pa = np.outer(np.arange(in_shape[0] * in_shape[1]), np.ones(in_shape[2]), ).flatten()
        cols_pa = np.arange(in_shape[0] * in_shape[1] * in_shape[2]).flatten()
        rows_pb = rows_pa
        cols_pb = np.repeat(np.arange(in_shape[2]*in_shape[0]).reshape(in_shape[0],-1), repeats=in_shape[1], axis=0).flatten()
               
        # print('rows\n', rows.shape)
        # print('cols\n', cols.shape)
        self.declare_derivatives(out_name, in_name_1, rows=rows_pa, cols=cols_pa)
        self.declare_derivatives(out_name, in_name_2, rows=rows_pb, cols=cols_pb)
        # print('finish define----------------')

    def compute(self, inputs, outputs):
        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        out_name = self.parameters["out_name"]

        outputs[out_name] = np.einsum(
            'kij,kj->ki',
            inputs[in_name_1],
            inputs[in_name_2],
        )
        # print(outputs[out_name])

    def compute_derivatives(self, inputs, derivatives):
        in_name_1 = self.parameters["in_name_1"]
        in_name_2 = self.parameters["in_name_2"]
        out_name = self.parameters["out_name"]
        in_shape = self.parameters["in_shape"]

        derivatives[out_name, in_name_1] = np.repeat(inputs[in_name_2],repeats=in_shape[1],axis=0).flatten()
        
        derivatives[out_name, in_name_2] = inputs[in_name_1].flatten()

if __name__ == "__main__":
    import csdl_om
    k = 4
    i = 3
    j = 8
    in_shape = (k, i, j)
    in_name_1 = 'in_1'
    in_name_2 = 'in_2'
    out_name = 'out'

    in_1_val = np.random.random((k, i, j))
    in_2_val = np.random.random((k, j))
    model = csdl.Model()
    a = model.declare_variable(in_name_1, val=in_1_val)
    b = model.declare_variable(in_name_2, val=in_2_val)
    product = csdl.custom(a,
                          b,
                          op=EinsumKijKjKi(in_name_1=in_name_1,
                                            in_name_2=in_name_2,
                                            in_shape=in_shape,
                                            out_name=out_name))

    model.register_output(out_name, product)
    sim = csdl_om.Simulator(model)

    sim.run()
    sim.check_partials(compact_print=True)
