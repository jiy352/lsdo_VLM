import numpy as np
from openmdao.components.interp_util.interp import InterpND

import csdl
# import csdl_om
from VLM_package.VLM_preprocessing.utils.generate_simple_mesh import *
from ufl import Or

# class BSpline(csdl.CustomExplicitOperation):
#     def initialize(self):
#         print('initialize')
#         """
#         Declare parameters.
#         twist_cp unit: degree
#         """
#         self.parameters.declare("surface_shapes")
#         self.parameters.declare("surface_names")

#     def define(self):
#         surface_shapes = self.parameters["surface_shapes"]
#         surface_names = self.parameters["surface_names"]

#         for i in len(surface_shapes):
#             ny = surface_shapes[i][2]
#             surface_name = surface_names[i]
#             self.add_output(surface_name + 'twist', shape=(ny, ))
#             self.add_input(surface_name + 'twist_cp', shape=(3, ))
#             self.declare_derivatives(surface_name + 'twist',
#                                      surface_name + 'twist_cp')

#     def compute(self, inputs, outputs):
#         surface_shapes = self.parameters["surface_shapes"]
#         surface_names = self.parameters["surface_names"]

#         for i in len(surface_shapes):
#             surface_name = surface_names[i]
#             ny = surface_shapes[i][2]
#             x_interp = np.linspace(0.0, 1.0, int(ny))
#             interp = InterpND(method='bsplines',
#                               num_cp=3,
#                               x_interp=x_interp,
#                               order=3)
#             outputs[surface_name + 'twist'] = interp.evaluate_spline(
#                 inputs[surface_name + 'twist_cp'], compute_derivative=False)

#     def compute_derivatives(self, inputs, derivatives):
#         surface_shapes = self.parameters["surface_shapes"]
#         surface_names = self.parameters["surface_names"]

#         for i in len(surface_shapes):
#             surface_name = surface_names[i]
#             ny = surface_shapes[i][2]
#             x_interp = np.linspace(0.0, 1.0, int(ny))
#             interp = InterpND(method='bsplines',
#                               num_cp=3,
#                               x_interp=x_interp,
#                               order=3)
#             _, dy = interp.evaluate_spline(inputs[surface_name + 'twist_cp'],
#                                            compute_derivative=True)
#             derivatives[surface_name + 'twist', surface_name + 'twist_cp'] = dy


class BSpline(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare("ny", default=5)

    def define(self):
        ny = self.parameters['ny']
        self.add_output('twist', shape=(ny, ))
        self.add_input('twist_cp', shape=(3, ))
        self.declare_derivatives('twist', 'twist_cp')

    def compute(self, inputs, outputs):
        ny = self.parameters['ny']

        x_interp = np.linspace(0.0, 1.0, int(ny))
        interp = InterpND(method='bsplines',
                          num_cp=3,
                          x_interp=x_interp,
                          order=3)
        outputs['twist'] = interp.evaluate_spline(inputs['twist_cp'],
                                                  compute_derivative=False)

    def compute_derivatives(self, inputs, derivatives):

        ny = self.parameters['ny']

        x_interp = np.linspace(0.0, 1.0, int(ny))
        interp = InterpND(method='bsplines',
                          num_cp=3,
                          x_interp=x_interp,
                          order=3)

        _, dy = interp.evaluate_spline(inputs['twist_cp'],
                                       compute_derivative=True)
        derivatives['twist', 'twist_cp'] = dy


# NY = 7
# NX = 5

# symmetry = False
# mesh = generate_simple_mesh(NX, NY)
# surface_shapes = [(1, NX, NY, 3)]
# surface_names = ['wing']
# val = np.zeros(NY)
# top_model = csdl.Model()

# twist_cp = top_model.declare_variable('twist_cp', val=np.random.random(3))
# # twist_cp = top_model.declare_variable('wingtwist_cp', val=np.random.random(3))
# spline = csdl.custom(twist_cp, op=BSpline())
# # spline = csdl.custom(twist_cp, op=BSpline(surface_shapes, surface_names))
# top_model.register_output('b-spline', spline)
# sim = csdl_om.Simulator(top_model)

# sim.run()

# sim.prob.check_partials(compact_print=True, abs_err_tol=1e-5, rel_err_tol=1e-5)