from scrublet.helper_functions import *
from sklearn.decomposition import PCA, TruncatedSVD
import pandas as pd
import logging
from .utilities import calculate_posterior_mc_rate, highly_variable_methylation_feature

class Scrublet():
    def __init__(self, mc, tc, bins, sim_doublet_ratio=2.0, n_neighbors=None, expected_doublet_rate=0.1, stdev_doublet_rate=0.02):
        # initialize counts matrices
        self._M_obs = mc
        self._T_obs = tc
        rateb = calculate_posterior_mc_rate(mc, tc)
        disp = highly_variable_methylation_feature(rateb, np.mean(tc, axis=0), bins)
        idx = np.argsort(disp)[::-1]
        self._hvg_filter = idx[:2000]
        self._E_obs = rateb[:, self._hvg_filter]
        self._E_sim = None
        self._embeddings = {}
        self.sim_doublet_ratio = sim_doublet_ratio
        self.n_neighbors = n_neighbors
        self.expected_doublet_rate = expected_doublet_rate
        self.stdev_doublet_rate = stdev_doublet_rate
        if self.n_neighbors is None:
            self.n_neighbors = int(round(0.5*np.sqrt(self._E_obs.shape[0])))

    ######## Core Scrublet functions ########

    def scrub_doublets(self, synthetic_doublet_umi_subsampling=1.0, use_approx_neighbors=True, distance_metric='euclidean', get_doublet_neighbor_parents=False, min_counts=3, min_cells=3, min_gene_variability_pctl=85, log_transform=False, mean_center=True, normalize_variance=True, n_prin_comps=30, verbose=True):
        t0 = time.time()

        print_optional('Simulating doublets...', verbose)
        self.simulate_doublets(sim_doublet_ratio=self.sim_doublet_ratio, synthetic_doublet_umi_subsampling=synthetic_doublet_umi_subsampling)

        print_optional('Embedding transcriptomes using PCA...', verbose)
        self.pipeline_pca(n_prin_comps=n_prin_comps)

        print_optional('Calculating doublet scores...', verbose)
        self.calculate_doublet_scores(use_approx_neighbors=use_approx_neighbors, distance_metric=distance_metric, get_doublet_neighbor_parents=get_doublet_neighbor_parents)
        self.call_doublets(verbose=verbose)

        t1=time.time()
        print_optional('Elapsed time: {:.1f} seconds'.format(t1 - t0), verbose)
        return self.doublet_scores_obs_, self.predicted_doublets_, self.doublet_scores_sim_

    def simulate_doublets(self, sim_doublet_ratio=None, synthetic_doublet_umi_subsampling=1.0):
        if sim_doublet_ratio is None:
            sim_doublet_ratio = self.sim_doublet_ratio
        else:
            self.sim_doublet_ratio = sim_doublet_ratio

        n_obs = self._E_obs.shape[0]
        n_sim = int(n_obs * sim_doublet_ratio)
        pair_ix = np.random.randint(0, n_obs, size=(n_sim, 2))
        
        M1 = self._M_obs[pair_ix[:,0],:]
        M2 = self._M_obs[pair_ix[:,1],:]
        T1 = self._T_obs[pair_ix[:,0],:]
        T2 = self._T_obs[pair_ix[:,1],:]

        self._E_sim = calculate_posterior_mc_rate(M1+M2, T1+T2)[:, self._hvg_filter]
        self.doublet_parents_ = pair_ix
        return

    def pipeline_pca(self, n_prin_comps=50):
        X_obs = self._E_obs
        X_sim = self._E_sim
        pca = PCA(n_components=n_prin_comps).fit(X_obs)
        self.manifold_obs_ = pca.transform(X_obs)
        self.manifold_sim_ = pca.transform(X_sim)
        return

    def calculate_doublet_scores(self, use_approx_neighbors=True, distance_metric='euclidean', get_doublet_neighbor_parents=False):
        self._nearest_neighbor_classifier(
            k=self.n_neighbors,
            exp_doub_rate=self.expected_doublet_rate,
            stdev_doub_rate=self.stdev_doublet_rate,
            use_approx_nn=use_approx_neighbors, 
            distance_metric=distance_metric,
            get_neighbor_parents=get_doublet_neighbor_parents
            )
        return self.doublet_scores_obs_

    def _nearest_neighbor_classifier(self, k=40, use_approx_nn=True, distance_metric='euclidean', exp_doub_rate=0.1, stdev_doub_rate=0.03, get_neighbor_parents=False):
        manifold = np.vstack((self.manifold_obs_, self.manifold_sim_))
        doub_labels = np.concatenate((np.zeros(self.manifold_obs_.shape[0], dtype=int), 
                                      np.ones(self.manifold_sim_.shape[0], dtype=int)))

        n_obs = np.sum(doub_labels == 0)
        n_sim = np.sum(doub_labels == 1)
        
        # Adjust k (number of nearest neighbors) based on the ratio of simulated to observed cells
        k_adj = int(round(k * (1+n_sim/float(n_obs))))
        
        # Find k_adj nearest neighbors
        neighbors = get_knn_graph(manifold, k=k_adj, dist_metric=distance_metric, approx=use_approx_nn, return_edges=False)
        
        # Calculate doublet score based on ratio of simulated cell neighbors vs. observed cell neighbors
        doub_neigh_mask = doub_labels[neighbors] == 1
        n_sim_neigh = doub_neigh_mask.sum(1)
        n_obs_neigh = doub_neigh_mask.shape[1] - n_sim_neigh
        
        rho = exp_doub_rate
        r = n_sim / float(n_obs)
        nd = n_sim_neigh.astype(float)
        ns = n_obs_neigh.astype(float)
        N = float(k_adj)
        
        # Bayesian
        q=(nd+1)/(N+2)
        Ld = q*rho/r/(1-rho-q*(1-rho-rho/r))

        se_q = np.sqrt(q*(1-q)/(N+3))
        se_rho = stdev_doub_rate

        se_Ld = q*rho/r / (1-rho-q*(1-rho-rho/r))**2 * np.sqrt((se_q/q*(1-rho))**2 + (se_rho/rho*(1-q))**2)

        self.doublet_scores_obs_ = Ld[doub_labels == 0]
        self.doublet_scores_sim_ = Ld[doub_labels == 1]
        self.doublet_errors_obs_ = se_Ld[doub_labels==0]
        self.doublet_errors_sim_ = se_Ld[doub_labels==1]

        # get parents of doublet neighbors, if requested
        neighbor_parents = None
        if get_neighbor_parents:
            parent_cells = self.doublet_parents_
            neighbors = neighbors - n_obs
            neighbor_parents = []
            for iCell in range(n_obs):
                this_doub_neigh = neighbors[iCell,:][neighbors[iCell,:] > -1]
                if len(this_doub_neigh) > 0:
                    this_doub_neigh_parents = np.unique(parent_cells[this_doub_neigh,:].flatten())
                    neighbor_parents.append(this_doub_neigh_parents)
                else:
                    neighbor_parents.append([])
            self.doublet_neighbor_parents_ = np.array(neighbor_parents)
        return
    
    def call_doublets(self, threshold=None, verbose=True):
        if threshold is None:
            # automatic threshold detection
            # http://scikit-image.org/docs/dev/api/skimage.filters.html
            from skimage.filters import threshold_minimum
            try:
                threshold = threshold_minimum(self.doublet_scores_sim_)
                if verbose:
                    print("Automatically set threshold at doublet score = {:.2f}".format(threshold))
            except:
                self.predicted_doublets_ = None
                if verbose:
                    print("Warning: failed to automatically identify doublet score threshold. Run `call_doublets` with user-specified threshold.")
                return self.predicted_doublets_

        Ld_obs = self.doublet_scores_obs_
        Ld_sim = self.doublet_scores_sim_
        se_obs = self.doublet_errors_obs_
        Z = (Ld_obs - threshold) / se_obs
        self.predicted_doublets_ = Ld_obs > threshold
        self.z_scores_ = Z
        self.threshold_ = threshold
        self.detected_doublet_rate_ = (Ld_obs>threshold).sum() / float(len(Ld_obs))
        self.detectable_doublet_fraction_ = (Ld_sim>threshold).sum() / float(len(Ld_sim))
        self.overall_doublet_rate_ = self.detected_doublet_rate_ / self.detectable_doublet_fraction_

        if verbose:
            print('Detected doublet rate = {:.1f}%'.format(100*self.detected_doublet_rate_))
            print('Estimated detectable doublet fraction = {:.1f}%'.format(100*self.detectable_doublet_fraction_))
            print('Overall doublet rate:')
            print('\tExpected   = {:.1f}%'.format(100*self.expected_doublet_rate))
            print('\tEstimated  = {:.1f}%'.format(100*self.overall_doublet_rate_))
            
        return self.predicted_doublets_


