import numpy as np
import ssm

from jPCA import JPCA
from jPCA.util import ensure_datas_is_list, preprocess

__all__ = ['PJPCA']


class PJPCA(JPCA):
    """
    A probabilistic version of jPCA. Internally, we use SSM
    (https://github.com/slinderman/ssm) to fit an LDS where the dynamics matrix
    is constrained to be orthogonal. The LDS can have different emissions types
    depending on the data: e.g poisson_orthogogonal emissions for single-trial
    data and gaussian_orthogonal emissions for trial-averaged and smoothed data.
    """
    def __init__(self, num_neurons,
                 emissions="gaussian_orthog",
                 dynamics="rotational",
                 num_jpcs=6):
        assert num_jpcs % 2 == 0, "num_jpcs must be an even number."
        self.num_jpcs = num_jpcs
        self.jpcs = None
        self.lds = ssm.LDS(num_neurons,
            num_jpcs, dynamics=dynamics, emissions=emissions)
        if "orthog" in emissions:
            self.alpha = 0.0 
        else:
            self.alpha = 0.5

    def project_covs(self, covs):
        """ Project the covariances onto the jPCs for visualization

        Returns
        -------
        covs: list of num_jpcs x num_jpcs covariance matrices, with state uncertainty at each
              timestep.
        """
        assert self.jpcs is not None, "Model must be fit before calling project_covs."
        return [self.jpcs @ cov @ self.jpcs.T for cov in covs]
        
    @ensure_datas_is_list
    def fit(self,
            datas,
            num_iters=50,
            subtract_cc_mean=False,
            tstart=0,
            tend=-1,
            times=None,
            align_axes_to_data=True,
            soft_normalize=5,
            **fit_kwargs):

        assert isinstance(datas, list), "datas must be a list."
        assert datas, "datas cannot be empty"
        T = datas[0].shape[0]
        if times is None:
            times = np.arange(T)
            tstart = 0
            tend = times[-1]

        # We don't use PCA with the probabilistic version -- the emissions model
        # does that for us.
        processed_datas, full_data_var, _ = \
            preprocess(datas, times, tstart=tstart, tend=tend, pca=False,
                       subtract_cc_mean=subtract_cc_mean, num_pcs=-1,
                       soft_normalize=soft_normalize)
        self.full_data_var = full_data_var

        # Fit the LDS
        elbo, posterior = self.lds.fit(processed_datas, num_iters=num_iters,
            alpha=self.alpha, **fit_kwargs)

        # Set jPCS using recovered dynamics matrix.
        self.jpcs = self._calculate_jpcs(self.lds.dynamics.A)

        mus = posterior.mean_continuous_states
        sigmas = [exp[2] for exp in posterior.continuous_expectations]

        if align_axes_to_data:
            self.align_jpcs(mus)

        projected_mus, _ = self.project_states(mus)
        projected_sigmas = self.project_covs(sigmas)

        return elbo, posterior, projected_mus, projected_sigmas


