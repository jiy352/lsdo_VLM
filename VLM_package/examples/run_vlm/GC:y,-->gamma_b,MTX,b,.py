

# GC:y,-->gamma_b,MTX,b,
import scipy.sparse as sp
import numpy as np

# GC:y,-->gamma_b,MTX,b,

# :::::::y-->gamma_b,MTX,b,:::::::

# _00ed linear_combination_eval_py_pb
path_to__00ec = py_p_00ec.copy()
path_to_b = py_pb.copy()

# _00eb einsum_eval_p_00ec_pgamma_b
_00eb_temp_einsum = _00eb_partial_func(MTX, gamma_b)
p_00ec_pMTX = _00eb_temp_einsum[0]
p_00ec_pgamma_b = _00eb_temp_einsum[1]
path_to_MTX = path_to__00ec@p_00ec_pMTX
path_to_gamma_b = path_to__00ec@p_00ec_pgamma_b
dy_dgamma_b = path_to_gamma_b.copy()
dy_dMTX = path_to_MTX.copy()
dy_db = path_to_b.copy()