# from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np
from numpy.core.fromnumeric import size


class MeshParameterizationComp(Model):
    """
    Compute pointsets for lifting surface meshes given chord and span parameters
    parameters
    ----------
    root_chord_l
    taper_ratio
    span_l
    offsets

    Returns
    -------
    1. mesh[num_nodes,num_pts_chord, num_pts_span, 3] : csdl array
    lifting surface mesh pointsets
    """
    def initialize(self):
        self.parameters.declare('surface_names', types=list)
        self.parameters.declare('surface_shapes', types=list)
        self.parameters.declare('mesh_unit', default='m')

    def define(self):
        # load options
        surface_names = self.parameters['surface_names']
        surface_shapes = self.parameters['surface_shapes']
        mesh_unit = self.parameters['mesh_unit']
        num_nodes = surface_shapes[0][0]

        system_size = sum((i[1] - 1) * (i[2] - 1) for i in surface_shapes)
        def_mesh_list = []

        # loop through lifting surfaces to compute outputs
        start = 0
        for i in range(len(surface_names)):
            # load name of the geometry mesh, number of points in chord and spanwise direction
            surface_name = surface_names[i]
            num_pts_chord = surface_shapes[i][1]
            num_pts_span = surface_shapes[i][2]
            nx = num_pts_chord
            ny = num_pts_span

            delta = (num_pts_chord - 1) * (num_pts_span - 1)
            chord_l_name = surface_name + '_chord_l'
            span_l_name = surface_name + '_span_l'
            tx_name = surface_name + '_tx'
            ty_name = surface_name + '_ty'

            # get names of the outputs:
            # 1. bd_vtx_coords_name, 2. coll_pts_coords_name, 3. chord_name, 4. span_name, 5. s_panel_name


            chord = self.declare_variable(chord_l_name,shape=(1,))
            span = self.declare_variable(span_l_name,shape=(1,))


            tensor_x = np.outer(np.arange(nx), np.ones(ny)).reshape(1,nx,ny,1)
            tensor_y  = np.outer(np.arange(ny), np.ones(nx)).T.reshape(1,nx,ny,1)

            # print('tensor_x/(nx-1)',tensor_x/(nx-1))
            # print('tensor_x/(nx-1) * chord',tensor_x/(nx-1) * chord)

            tensor_x_csdl = self.declare_variable(tx_name,val=tensor_x/(nx-1))
            tensor_y_csdl = self.declare_variable(ty_name,val=tensor_y/(ny-1))

            taper_ratio = self.declare_variable("taper_ratio",val=0.5)

            # x = tensor_x/(nx-1) * chord
            # y = tensor_y/(nx-1) * span
            x = tensor_x_csdl * csdl.expand(chord,shape=tensor_x_csdl.shape)
            y = tensor_y_csdl * csdl.expand(span,shape=tensor_y_csdl.shape)

            # self.register_output('x_name',x+0)
            # self.register_output('y_name',y+0)
            mesh = self.create_output(surface_name,val=np.zeros((1,nx,ny,3)))
            mesh[:,:,:,1] = y

            h_over_H = y/csdl.expand(span,y.shape,"l->ijkl")
            # self.print_var(h_over_H)

            ones = self.create_input("dummy_ones",np.ones(y.shape))


            taper_tensor = ones-h_over_H+csdl.expand(taper_ratio, y.shape, "l->ijkl")*h_over_H
            # self.print_var(csdl.expand(taper_ratio, y.shape, "l->ijkl")*h_over_H)
            # self.print_var(-h_over_H+csdl.expand(taper_ratio, y.shape, "l->ijkl")*h_over_H)
            mesh[:,:,:,0] = (x-csdl.expand(chord/2,x.shape,"l->ijkl"))*taper_tensor

            self.register_output("taper_tensor", taper_tensor)




if __name__ == "__main__":
    import python_csdl_backend

    simulator_name = 'python_csdl_backend'

    num_nodes = 2
    num_pts_chord = 3
    num_pts_span = 4


    surface_names = ['wing_1', 'wing_2']
    surface_shapes = [(num_nodes, num_pts_chord, num_pts_span, 3),
                      (num_nodes, num_pts_chord + 1, num_pts_span + 1, 3)]
    model_1 = Model()


    wing_1_chord = model_1.create_input('wing_1_chord_l', val=2)
    wing_2_chord = model_1.create_input('wing_2_chord_l', val=2)

    wing_1_span = model_1.create_input('wing_1_span_l', val=5)
    wing_2_span = model_1.create_input('wing_2_span_l', val=5)

    model_1.add(MeshParameterizationComp(surface_names=surface_names,
                                      surface_shapes=surface_shapes),
                name='MeshParameterizationComp')


    if simulator_name == 'python_csdl_backend':
        sim = python_csdl_backend.Simulator(model_1)

        sim.run()
        # sim.check_partials(compact_print=True)
