import enum
import numpy as np

# class AcStates_vlm(enum.Enum):
#     u = 'AcBfState_u'
#     v = 'AcBfState_v'
#     w = 'AcBfState_w'
#     p = 'AcBfState_p'
#     q = 'AcBfState_q'
#     r = 'AcBfState_r'
#     phi = 'AcBfState_phi'
#     theta = 'AcBfState_theta'
#     psi = 'AcBfState_psi'
#     x = 'AcBfState_x'
#     y = 'AcBfState_y'
#     z = 'AcBfState_z'
#     phiw = 'AcWState_phi_w'
#     gamma = 'AcWState_gamma'
#     psiw = 'AcWState_psi_w'
#     rho = 'AcWState_rho'


class AcStates_vlm(enum.Enum):
    u = 'u'
    v = 'v'
    w = 'w'
    p = 'p'
    q = 'q'
    r = 'r'
    phi = 'phi'
    theta = 'theta'
    psi = 'psi'
    x = 'x'
    y = 'y'
    z = 'z'
    phiw = 'phiw'
    gamma = 'gamma'
    psiw = 'psiw'
    rho = 'rho'


# num_nodes = 3

# v_inf = np.array([50, 50, 50])
# alpha_deg = np.array([2, 4, 6])
# alpha = alpha_deg / 180 * np.pi
# vx = -v_inf * np.cos(alpha)
# vz = -v_inf * np.sin(alpha)

# AcStates_val_dict = {
#     AcStates_vlm.u.value: vx.reshape(num_nodes, 1),
#     AcStates_vlm.v.value: np.zeros((num_nodes, 1)),
#     AcStates_vlm.w.value: vz.reshape(num_nodes, 1),
#     AcStates_vlm.p.value: np.zeros((num_nodes, 1)),
#     AcStates_vlm.q.value: np.zeros((num_nodes, 1)),
#     AcStates_vlm.r.value: np.zeros((num_nodes, 1)),
#     AcStates_vlm.phi.value: np.zeros((num_nodes, 1)),
#     AcStates_vlm.theta.value: np.zeros((num_nodes, 1)),
#     AcStates_vlm.psi.value: np.zeros((num_nodes, 1)),
#     AcStates_vlm.x.value: np.zeros((num_nodes, 1)),
#     AcStates_vlm.y.value: np.zeros((num_nodes, 1)),
#     AcStates_vlm.z.value: np.zeros((num_nodes, 1)),
#     AcStates_vlm.phiw.value: np.ones((num_nodes, 1)),
#     AcStates_vlm.gamma.value: np.ones((num_nodes, 1)),
#     AcStates_vlm.psiw.value: np.ones((num_nodes, 1)),
#     AcStates_vlm.rho.value: np.ones((num_nodes, 1)) * 0.96,
# }

num_nodes = 1

v_inf = np.array([50])
alpha_deg = np.array([2])
alpha = alpha_deg / 180 * np.pi
vx = -v_inf * np.cos(alpha)
vz = -v_inf * np.sin(alpha)

AcStates_val_dict = {
    AcStates_vlm.u.value: vx.reshape(num_nodes, 1),
    AcStates_vlm.v.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.w.value: vz.reshape(num_nodes, 1),
    AcStates_vlm.p.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.q.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.r.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.phi.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.theta.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.psi.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.x.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.y.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.z.value: np.zeros((num_nodes, 1)),
    AcStates_vlm.phiw.value: np.ones((num_nodes, 1)),
    AcStates_vlm.gamma.value: np.ones((num_nodes, 1)),
    AcStates_vlm.psiw.value: np.ones((num_nodes, 1)),
    AcStates_vlm.rho.value: np.ones((num_nodes, 1)) * 0.96,
}

# class AcStates_vlm(enum.Enum):
#     u = 'AcBfState_u'
#     v = 'AcBfState_v'
#     w = 'AcBfState_w'
#     p = 'AcBfState_p'
#     q = 'AcBfState_q'
#     r = 'AcBfState_r'
#     phi = 'AcBfState_phi'
#     theta = 'AcBfState_theta'
#     psi = 'AcBfState_psi'
#     x = 'AcBfState_x'
#     y = 'AcBfState_y'
#     z = 'AcBfState_z'
#     phiw = 'AcWState_phi_w'
#     gamma = 'AcWState_gamma'
#     psiw = 'AcWState_psi_w'

# for data in AcStates_vlm:
#     print('{:15} = {}'.format(data.name, data.value))

# print('AcStates_vlm.u name is:', AcStates_vlm.u.name)
# print('AcStates_vlm.u value is:', AcStates_vlm.u.value)
