
from typing import OrderedDict
import numpy as np
import csdl
from VLM_package.vlm_solver import VLMSolverModel

def get_pressure_contour(AcStates_val_dict, camber_mesh_dict, eval_pts_dict, Simulator):

    model_1 = csdl.Model()

    ####################################################################
    # 2. add acstates
    ####################################################################

    for i in range(len(AcStates_val_dict)):
        name = list(AcStates_val_dict)[i]
        # print('{:15} = {}={}'.format(name, AcStates_val_dict[name]))
        
        variable = model_1.create_input(name,
                                        val=AcStates_val_dict[name])


    ####################################################################
    # 2. add VLM meshes
    ####################################################################
    surface_shapes = []
    surface_names = []

    for i in range(len(camber_mesh_dict)):
        surface_name = list(camber_mesh_dict)[i]
        surface = model_1.create_input(surface_name, val=camber_mesh_dict[surface_name])

        surface_names.append(list(camber_mesh_dict)[i])
        surface_shapes.append(camber_mesh_dict[surface_name].shape)

    ####################################################################
    # 3. add upper and lower surfaces
    ####################################################################
    eval_pts_shapes = []


    for i in range(len(eval_pts_dict)):
        eval_pts_name = list(eval_pts_dict)[i]
        eval_pts = model_1.create_input(eval_pts_name, val=eval_pts_dict[eval_pts_name])

        eval_pts_shapes.append(eval_pts_dict[eval_pts_name].shape)



    submodel = VLMSolverModel(
        surface_names=surface_names,
        surface_shapes=surface_shapes,
        num_nodes=num_nodes,
        # free_stream_velocities=free_stream_velocities,
        eval_pts_location=0.25,
        # The location of the evaluation point is on the quarter-chord,
        # if this is not provided, it is defaulted to be 0.25.
        eval_pts_option='user_defined',
        eval_pts_names= list(eval_pts_dict),
        eval_pts_shapes=eval_pts_shapes,
        AcStates='dummy',
        cl0=[0.]
    )
    model_1.add(submodel, 'VLMSolverModel')

    sim = Simulator(model_1)

    sim.run()
    pressure_contour_dict = OrderedDict()
    for i in range(len(eval_pts_dict)):
        eval_pts_name = list(eval_pts_dict)[i]
        pressure_contour_dict[eval_pts_name+'_pressure_coeff'] = sim[eval_pts_name+'_pressure_coeff']

    return pressure_contour_dict

if __name__ == "__main__":
    from VLM_package.examples.run_vlm.utils.generate_mesh import generate_mesh
    from python_csdl_backend import Simulator

    ####################################################################
    # 1. Define VLM inputs that share the common names within CADDEE
    ####################################################################


    num_nodes = 1

    # v_inf = np.array([50])
    alpha_deg = np.array([3.125382526501336])
    alpha = alpha_deg / 180 * np.pi

    # single lifting surface
    nx = 3  # number of points in streamwise direction
    ny = 11  # number of points in spanwise direction

    # surface_names = ['wing', 'wing_1']
    # surface_shapes = [(num_nodes, nx, ny, 3), (num_nodes, nx, ny, 3)]

    # surface_names = ['wing']
    # surface_shapes = [(num_nodes, nx, ny, 3)]

    chord = 1.49352
    span = 16.2 / chord
    # https://github.com/LSDOlab/nasa_uli_tc1/blob/222d877228b609076dd352945f4cfe2d158d4973/execution_scripts/c172_climb.py#L33

    mesh_dict = {
        "num_y": ny,
        "num_x": nx,
        "wing_type": "rect",
        "symmetry": False,
        "span": span,
        "root_chord": chord,
        "span_cos_spacing": False,
        "chord_cos_spacing": False,
    }

    # Generate mesh of a rectangular wing
    mesh = generate_mesh(mesh_dict) #(nx,ny,3)
    offset = span / 2

    mesh_val = np.zeros((num_nodes, nx, ny, 3))
    mesh_val_1 = np.zeros((num_nodes, nx, ny, 3))
    for i in range(num_nodes):
        mesh_val[i, :, :, :] = mesh
        mesh_val[i, :, :, 0] = mesh.copy()[:, :, 0]
        mesh_val[i, :, :, 1] = mesh.copy()[:, :, 1] + offset


    offset_1 = offset*3

    for i in range(num_nodes):
        mesh_val_1[i, :, :, :] = mesh.copy() + offset_1
        mesh_val_1[i, :, :, 0] = mesh.copy()[:, :, 0]
        mesh_val_1[i, :, :, 1] = mesh.copy()[:, :, 1] + offset_1


    # vals in the dict shape = (num_nodes, 1); num_nodes=1
    AcStates_val_dict = {
        'u': np.array([[50.39014388]]), # nonzeros
        'v': np.zeros((num_nodes, 1)),
        'w': np.array([[2.75142193]]),  # nonzeros
        'p': np.zeros((num_nodes, 1)),
        'q': np.zeros((num_nodes, 1)),
        'r': np.zeros((num_nodes, 1)),
        'phi': np.zeros((num_nodes, 1)),
        'theta': np.ones((num_nodes, 1))*alpha, # nonzeros (in rad)
        'psi': np.zeros((num_nodes, 1)),
        'x': np.zeros((num_nodes, 1)),
        'y': np.zeros((num_nodes, 1)),
        'z': np.zeros((num_nodes, 1)), # can be nonzeros
        'phiw': np.zeros((num_nodes, 1)),
        'gamma': np.zeros((num_nodes, 1)),
        'psiw': np.zeros((num_nodes, 1)),
    }

    camber_mesh_dict = {
        'wing_camber_left': mesh_val, # nonzeros
        # 'wing_camber_right': mesh_val_1,
        # 'tail_camber_left': np.zeros((num_nodes, 1)),
        # 'tail_camber_right': np.zeros((num_nodes, 1)),

    }

    wing_upper_surface = mesh_val.copy()
    wing_upper_surface[:,:,:,2] = mesh_val[:,:, :, 2].copy()+.5
    wing_upper_surface_coll = 0.5*(wing_upper_surface[:,:-1,:-1,:]+wing_upper_surface[:,1:,1:,:])

    wing_lower_surface = mesh_val.copy()
    wing_lower_surface[:,:,:,2] = mesh_val[:,:, :, 2].copy()-.5
    wing_lower_surface_coll = 0.5*(wing_lower_surface[:,:-1,:-1,:]+wing_lower_surface[:,1:,1:,:])

    eval_pts_dict = {
        'wing_upper_surface': wing_upper_surface_coll, # nonzeros
        'wing_lower_surface': wing_lower_surface_coll,
    }

    pressure_contour_dict = get_pressure_contour(AcStates_val_dict, camber_mesh_dict, eval_pts_dict, Simulator)
    # import pyvista as pv
    # pv.global_theme.axes.show = True
    # pv.global_theme.font.label_size = 1
    # x = mesh_val[0,:, :, 0]
    # y = mesh_val[0,:, :, 1]
    # z = mesh_val[0,:, :, 2]+0.5

    # grid = pv.StructuredGrid(x, y, z)    
    

    # eval_pts_name = list(eval_pts_dict)[0]
    # pressure = pressure_contour_dict[eval_pts_name+'_pressure_coeff'].reshape(nx-1, ny-1)
    

    # grid.cell_data["pressure"] = np.moveaxis( pressure.reshape(nx-1 , ny-1 ), 0, 1).reshape((nx-1)*(ny-1), )
    #     # grid_1.cell_data["panel_forces"] = np.moveaxis(
    #     #     sim["panel_forces"][0, 56:, :].reshape(nx - 1, ny - 1, 3), 0,
    #     #     1).reshape(56, 3)
    #     # grid.save("vtks/front_wing.vtk")
    #     # grid_1.save("vtks/tail_wing.vtk")
    # p = pv.Plotter()
    # p.add_mesh(grid,  show_edges=True, opacity=1)


    # x = mesh_val[0,:, :, 0]
    # y = mesh_val[0,:, :, 1]
    # z_1 = mesh_val[0,:, :, 2]-0.5

    # grid_1 = pv.StructuredGrid(x, y, z_1)  
    # eval_pts_name = list(eval_pts_dict)[1]
    # pressure = pressure_contour_dict[eval_pts_name+'_pressure_coeff'].reshape(nx-1, ny-1)
    

    # grid_1.cell_data["pressure"] = np.moveaxis( pressure.reshape(nx-1 , ny-1 ), 0, 1).reshape((nx-1)*(ny-1), )
    # # p.add_mesh(gridw, color="blue", show_edges=True, opacity=.5)
    # # p.add_mesh(grid_1, color="red", show_edges=True, opacity=.5)
    # # p.add_mesh(gridw_1, color="red", show_edges=True, opacity=.5)
    # p.add_mesh(grid_1,  show_edges=True, opacity=1)
    # p.camera.view_angle = 60.0
    # p.add_axes_at_origin(labels_off=True, line_width=5)

    # p.show()