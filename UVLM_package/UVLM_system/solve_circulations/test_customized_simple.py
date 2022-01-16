import numpy as np

import openmdao.api as om

from csdl import CustomExplicitOperation
import numpy as np
from csdl_om import Simulator
import csdl

# TODO: potentially complain about the CustomExplicitOperation not running without model.add


class CustomExplicitOperation(CustomExplicitOperation):
    def define(self):
        self.add_input('Cl')
        self.add_input('Cd')
        self.add_input('rho')
        self.add_input('V')
        self.add_input('S')
        self.add_output('L')
        self.add_output('D')

        # declare derivatives of all outputs wrt all inputs
        self.declare_derivatives('*', '*')

    # ...
    def compute(self, inputs, outputs):
        outputs['L'] = 1 / 2 * inputs['Cl'] * inputs['rho'] * inputs[
            'V']**2 * inputs['S']
        outputs['D'] = 1 / 2 * inputs['Cd'] * inputs['rho'] * inputs[
            'V']**2 * inputs['S']

    def compute_derivatives(self, inputs, derivatives):
        derivatives[
            'L', 'Cl'] = 1 / 2 * inputs['rho'] * inputs['V']**2 * inputs['S']
        derivatives[
            'L', 'rho'] = 1 / 2 * inputs['Cl'] * inputs['V']**2 * inputs['S']
        derivatives[
            'L',
            'V'] = inputs['Cl'] * inputs['rho'] * inputs['V'] * inputs['S']
        derivatives[
            'L', 'S'] = 1 / 2 * inputs['Cl'] * inputs['rho'] * inputs['V']**2

        derivatives[
            'D', 'Cd'] = 1 / 2 * inputs['rho'] * inputs['V']**2 * inputs['S']
        derivatives[
            'D', 'rho'] = 1 / 2 * inputs['Cd'] * inputs['V']**2 * inputs['S']
        derivatives[
            'D',
            'V'] = inputs['Cd'] * inputs['rho'] * inputs['V'] * inputs['S']
        derivatives[
            'D', 'S'] = 1 / 2 * inputs['Cd'] * inputs['rho'] * inputs['V']**2


# compile using Simulator imported from back end...
model = csdl.Model()
model.create_input('Cl', val=10)
model.add(CustomExplicitOperation(), 'custom_operation')
sim = Simulator(model)
sim.run()
