import numpy as np
from scipy.optimize import minimize


def skew_sym_regress(X, X_dot, tol=1e-4):
  """
  Original data tensor is C x L x N where N is number of Neurons, L is length of each trial
  and C is number of conditions. We stack this to get L*C x N array.

  Args
  ----
    X_dot: First difference of (reduced dimension) data. Shape is T x N
           
    X: reduced dimension data. Shape is T x N
  """
     
  # 1) Initialize h using the odd part of the least-squares solution.
  # 2) call scipy.optimize.minimize and pass in our starting h, and x_dot, 
  T, N = X.shape
  M_lstq, _, _, _ = np.linalg.lstsq(X, X_dot, rcond=None)
  M_lstq = M_lstq.T
  M_init = 0.5 * (M_lstq - M_lstq.T)
  h_init = _reshape_mat2vec(M_init, N)

  options=dict(maxiter=10000, gtol=tol)
  result = minimize(lambda h: _objective(h, X, X_dot),
                    h_init,
                    jac=lambda h: _grad_f(h, X, X_dot),
                    method='CG',
                    options=options)
  if not result.success:
    print("Optimization failed.")
    print(result.message)
  M = _reshape_vec2mat(result.x, N)
  assert(np.allclose(M, -M.T))
  return M


def _grad_f(h, X, X_dot):
  _, N = X.shape
  M = _reshape_vec2mat(h, N)
  dM = (X.T @ X @ M.T) - X.T @ X_dot
  return _reshape_mat2vec(dM.T - dM, N)


def _objective(h, X, X_dot):
  _, N = X.shape
  M = _reshape_vec2mat(h, N)
  return 0.5 * np.linalg.norm(X @ M.T - X_dot, ord='fro')**2


def _reshape_vec2mat(h, N):
  M = np.zeros((N, N))
  upper_tri_indices = np.triu_indices(N, k=1)
  M[upper_tri_indices] = h
  return M - M.T


def _reshape_mat2vec(M, N):
  upper_tri_indices = np.triu_indices(N, k=1)
  return M[upper_tri_indices]