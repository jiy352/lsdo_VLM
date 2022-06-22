import numpy as np
from scipy.sparse import coo_array


def compute_spars(surface_shapes):
    # nx = 3
    # ny = 3
    # nx_1 = 3
    # ny_1 = 4
    # # nx_2 = 3
    # # ny_2 = 3

    # surface_shapes = [
    #     (nx, ny, 3),
    #     (nx_1, ny_1, 3),
    #     # (nx_2, ny_2, 3),
    # ]
    num_total_bd_panel = 0
    # num_bd_panel = []
    num_bd_panel_array = np.array([])
    for i in range(len(surface_shapes)):
        surface_shape = surface_shapes[i]
        num_total_bd_panel += (surface_shape[1] - 1) * (surface_shape[2] - 1)

    num_total_bd_ind = np.arange(num_total_bd_panel)
    start = 0
    for i in range(len(surface_shapes)):
        surface_shape = surface_shapes[i]
        delta = (surface_shape[1] - 1) * (surface_shape[2] - 1)
        # num_bd_panel.append(num_total_bd_ind[start:start +
        #                                      delta][-(surface_shape[1] + 1):])
        # num_bd_panel.append(num_total_bd_ind[start:start +
        #                                      delta][-(surface_shape[1] - 1):])
        num_bd_panel_array = np.concatenate((
            num_bd_panel_array,
            num_total_bd_ind[start:start + delta][-(surface_shape[2] - 1):],
        ))
        start += delta
    '''this only works when there is only one row of wake panel streamwise
        can be generlized by given n_wake_pts_chord (num_wake_panel streamwise) as inputs'''
    num_wake_panel = num_bd_panel_array.size

    row = np.arange(num_wake_panel)
    col = num_bd_panel_array
    data = np.ones(num_wake_panel)
    sprs = coo_array(
        (data, (row, col)),
        shape=(num_wake_panel, num_total_bd_panel),
    )
    return sprs


if __name__ == "__main__":

    nx = 3
    ny = 3
    nx_1 = 3
    ny_1 = 4
    nx_2 = 3
    ny_2 = 3

    surface_shapes = [
        (nx, ny, 3),
        (nx_1, ny_1, 3),
        (nx_2, ny_2, 3),
    ]
    sprs = compute_spars(surface_shapes)