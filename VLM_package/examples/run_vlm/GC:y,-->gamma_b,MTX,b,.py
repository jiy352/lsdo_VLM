

# GC:y,-->gamma_b,MTX,b,
import scipy.sparse as sp
import numpy as np

# GC:y,-->gamma_b,MTX,b,

# :::::::y-->gamma_b,MTX,b,:::::::

# _00dE linear_combination eval_py_p_00dD

# _00dE linear_combination eval_py_pb
path_to__00dD = py_p_00dD.copy()
path_to_b = py_pb.copy()

# _00dC einsum eval_p_00dD_pMTX
p_00dD_pMTX = p_00dD_pMTX_func(MTX, gamma_b)

# _00dC einsum eval_p_00dD_pgamma_b
p_00dD_pgamma_b = p_00dD_pgamma_b_func(MTX, gamma_b)
path_to_MTX = path_to__00dD@p_00dD_pMTX
path_to_gamma_b = path_to__00dD@p_00dD_pgamma_b
dy_dgamma_b = path_to_gamma_b.copy()
dy_dMTX = path_to_MTX.copy()
dy_db = path_to_b.copy()