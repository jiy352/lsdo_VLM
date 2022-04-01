from VLM_package.vlm_solver_evel_pts import VLMSolverModel
import numpy as np

from VLM_package.VLM_preprocessing.generate_simple_mesh import *
from scipy.sparse import coo_array

nx = 3
ny = 19
surface_shapes = [(nx, ny, 3)]

v_inf = 50
alpha_deg = 10
alpha = alpha_deg / 180 * np.pi
vx = -v_inf * np.cos(alpha)
vz = -v_inf * np.sin(alpha)

model = Model()
mesh_val = generate_simple_mesh(nx, ny).reshape(1, nx, ny, 3)
eval_pts_location = 0.5
eval_pts_coords = ((1 - eval_pts_location) * 0.5 * mesh_val[:, 0:-1, 0:-1, :] +
                   (1 - eval_pts_location) * 0.5 * mesh_val[:, 0:-1, 1:, :] +
                   eval_pts_location * 0.5 * mesh_val[:, 1:, 0:-1, :] +
                   eval_pts_location * 0.5 * mesh_val[:, 1:, 1:, :])[0,
                                                                     0, :, :]

surface_names = ['wing']
free_stream_velocities = np.array([-vx, 0, -vz])

eval_pts_names = [x + '_eval_pts_coords' for x in surface_names]

eval_pts_shapes = [(x[0] - 2, x[1] - 1, 3)
                   for x in surface_shapes]  # leading edge
wing = model.create_input('wing', val=mesh_val.reshape(1, nx, ny, 3))
for i in range(len(eval_pts_names)):
    eval_pts = model.create_input(eval_pts_names[i],
                                  val=eval_pts_coords.reshape(
                                      eval_pts_shapes[i]))

row = np.arange(18)  #ny-1
col = np.arange(18)
data = np.ones(18)
sprs = [coo_array((data, (row, col)), shape=(18, 36))]
#here we need to define a sparse matrix, such that
# sprs@vector = the ones that we selected
# shapes: (num_evel_pts, num_total_circualtion_strength) num_total_circualtion_strength=(nx-1)*(ny-1)

eval_pts_ind = [np.arange(19)]
submodel = VLMSolverModel(
    surface_names=surface_names,
    surface_shapes=surface_shapes,
    free_stream_velocities=free_stream_velocities,
    eval_pts_shapes=eval_pts_shapes,
    eval_pts_ind=eval_pts_ind,
    sprs=sprs,
)
model.add(submodel, 'VLMSolverModel')

sim = Simulator(model)

sim.run()
print('lift', sim.prob['L'])
print('drag', sim.prob['D'])
sim.visualize_implementation()
