

# GC:y,-->gamma_b,MTX,b,
import scipy.sparse as sp
import numpy as np

# GC:y,-->gamma_b,MTX,b,

# :::::::y-->gamma_b,MTX,b,:::::::

# _00cG linear_combination eval_py_p_00cF

# _00cG linear_combination eval_py_pb
path_to__00cF = py_p_00cF.copy()
path_to_b = py_pb.copy()

# _00cE einsum eval_p_00cF_pMTX
p_00cF_pMTX = p_00cF_pMTX_func(MTX, gamma_b)

# _00cE einsum eval_p_00cF_pgamma_b
p_00cF_pgamma_b = p_00cF_pgamma_b_func(MTX, gamma_b)
path_to_MTX = path_to__00cF@p_00cF_pMTX
path_to_gamma_b = path_to__00cF@p_00cF_pgamma_b
dy_dgamma_b = path_to_gamma_b.copy()
dy_dMTX = path_to_MTX.copy()
dy_db = path_to_b.copy()