from apply_uq import apply_uq
import csdl_om
import csdl
import numpy as np

num_nodes = 5
x1_quad = np.array([[-0.42848501, 0.32218691,  1.,          1.67781309,  2.42848501]]).T
x2_quad = np.array([[-0.85697001,  0.6443738,  2.,          3.35562618, 4.85697001]]).T


class Run(csdl.Model):

    def define(self):
        x1 = self.create_input('x1', val=x1_quad)
        x2 = self.create_input('x2', val=x2_quad)

        y1 = csdl.cos(x1)
        y2 = csdl.exp(y1)
        y3 = -2*x2
        y4 = csdl.exp(y3)
        # y5 = csdl.reshape(y5, -1)
        # y6 = csdl.reshape(y6, -1)
        y7 = y4 + y2
        f = y7 + 3  # Returns f with size of k^2

        self.register_output('f', f)


model = Run()
apply_uq(model, num_nodes, ['x1', 'x2'])

sim = csdl_om.Simulator(model)
# sim.visualize_implementation()
sim.run()
# sim.run()

print('x1 = \n', sim['x1'])
print('x2 = \n', sim['x2'])
print('f = \n', sim['f'])

# print('---')
# print(sim['_0005'])
# print()
# print(sim['_0005_tiled'])
# # OUTPUT:
# [11.03410843  5.75893546  5.50163536  5.48453686  5.48338016 11.13273947
#  5.85756649  5.60026639  5.58316789  5.58201119 10.26731441  4.99214144
#  4.73484134  4.71774284  4.71658613  9.44948277  4.1743098   3.9170097
#  3.8999112   3.8987545   9.02017343  3.74500046  3.48770036  3.47060186
#  3.46944516]
