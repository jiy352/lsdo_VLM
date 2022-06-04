import enum
import numpy as np


class AcStates(enum.Enum):
    u = 'AcBfState_u'
    v = 'AcBfState_v'
    w = 'AcBfState_w'
    p = 'AcBfState_p'
    q = 'AcBfState_q'
    r = 'AcBfState_r'
    phi = 'AcBfState_phi'
    theta = 'AcBfState_theta'
    psi = 'AcBfState_psi'
    x = 'AcBfState_x'
    y = 'AcBfState_y'
    z = 'AcBfState_z'
    phiw = 'AcWState_phi_w'
    gamma = 'AcWState_gamma'
    psiw = 'AcWState_psi_w'
    rho = 'AcWState_rho'


# num_nodes = 3

# v_inf = np.array([50, 50, 50])
# alpha_deg = np.array([2, 4, 6])
# alpha = alpha_deg / 180 * np.pi
# vx = -v_inf * np.cos(alpha)
# vz = -v_inf * np.sin(alpha)

# AcStates_val_dict = {
#     AcStates.u.value: vx.reshape(num_nodes, 1),
#     AcStates.v.value: np.zeros((num_nodes, 1)),
#     AcStates.w.value: vz.reshape(num_nodes, 1),
#     AcStates.p.value: np.zeros((num_nodes, 1)),
#     AcStates.q.value: np.zeros((num_nodes, 1)),
#     AcStates.r.value: np.zeros((num_nodes, 1)),
#     AcStates.phi.value: np.zeros((num_nodes, 1)),
#     AcStates.theta.value: np.zeros((num_nodes, 1)),
#     AcStates.psi.value: np.zeros((num_nodes, 1)),
#     AcStates.x.value: np.zeros((num_nodes, 1)),
#     AcStates.y.value: np.zeros((num_nodes, 1)),
#     AcStates.z.value: np.zeros((num_nodes, 1)),
#     AcStates.phiw.value: np.ones((num_nodes, 1)),
#     AcStates.gamma.value: np.ones((num_nodes, 1)),
#     AcStates.psiw.value: np.ones((num_nodes, 1)),
#     AcStates.rho.value: np.ones((num_nodes, 1)) * 0.96,
# }

num_nodes = 1

v_inf = np.array([50])
alpha_deg = np.array([2])
alpha = alpha_deg / 180 * np.pi
vx = -v_inf * np.cos(alpha)
vz = -v_inf * np.sin(alpha)

AcStates_val_dict = {
    AcStates.u.value: vx.reshape(num_nodes, 1),
    AcStates.v.value: np.zeros((num_nodes, 1)),
    AcStates.w.value: vz.reshape(num_nodes, 1),
    AcStates.p.value: np.zeros((num_nodes, 1)),
    AcStates.q.value: np.zeros((num_nodes, 1)),
    AcStates.r.value: np.zeros((num_nodes, 1)),
    AcStates.phi.value: np.zeros((num_nodes, 1)),
    AcStates.theta.value: np.zeros((num_nodes, 1)),
    AcStates.psi.value: np.zeros((num_nodes, 1)),
    AcStates.x.value: np.zeros((num_nodes, 1)),
    AcStates.y.value: np.zeros((num_nodes, 1)),
    AcStates.z.value: np.zeros((num_nodes, 1)),
    AcStates.phiw.value: np.ones((num_nodes, 1)),
    AcStates.gamma.value: np.ones((num_nodes, 1)),
    AcStates.psiw.value: np.ones((num_nodes, 1)),
    AcStates.rho.value: np.ones((num_nodes, 1)) * 0.96,
}

# class AcStates(enum.Enum):
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

# for data in AcStates:
#     print('{:15} = {}'.format(data.name, data.value))

# print('AcStates.u name is:', AcStates.u.name)
# print('AcStates.u value is:', AcStates.u.value)
