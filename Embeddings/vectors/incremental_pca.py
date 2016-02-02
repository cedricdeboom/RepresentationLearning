"""
Port from sci-kit learn Github, adapted for sparse matrices
"""

import numpy as np
from scipy import linalg

from sklearn.decomposition.base import _BasePCA
from sklearn.utils import gen_batches
from sklearn.utils.extmath import svd_flip, _batch_mean_variance_update


class IncrementalPCA(_BasePCA):

    def __init__(self, n_components=None, whiten=False, copy=True,
                 batch_size=None):
        self.n_components = n_components
        self.whiten = whiten
        self.copy = copy
        self.batch_size = batch_size

    def fit(self, X, y=None):
        """Fit the model with X, using minibatches of size batch_size.
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        y: Passthrough for ``Pipeline`` compatibility.
        Returns
        -------
        self: object
            Returns the instance itself.
        """
        self.components_ = None
        self.mean_ = None
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.noise_variance_ = None
        self.var_ = None
        self.n_samples_seen_ = 0
        #X = check_array(X, dtype=np.float)        # --- ADJUSTED
        n_samples, _ = X.shape                     # --- ADJUSTED
        self.batch_size_ = self.batch_size         # --- ADJUSTED

        iteration = 0                                         # --- ADJUSTED
        for batch in gen_batches(n_samples, self.batch_size_):
            print "Iteration " + str(iteration)               # --- ADJUSTED
            self.partial_fit(X[batch].todense()[:, 0::4])   # --- ADJUSTED
            iteration += 1                                    # --- ADJUSTED
        return self

    def partial_fit(self, X, y=None):
        """Incremental fit with X. All of X is processed as a single batch.
        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        -------
        self: object
            Returns the instance itself.
        """
        #X = check_array(X, copy=self.copy, dtype=np.float)  # --- ADJUSTED
        X = np.asarray(X)                                    # --- ADJUSTED
        n_samples, n_features = X.shape
        if not hasattr(self, 'components_'):
            self.components_ = None

        if self.n_components is None:
            self.n_components_ = n_features
        elif not 1 <= self.n_components <= n_features:
            raise ValueError("n_components=%r invalid for n_features=%d, need "
                             "more rows than columns for IncrementalPCA "
                             "processing" % (self.n_components, n_features))
        else:
            self.n_components_ = self.n_components

        if (self.components_ is not None) and (self.components_.shape[0]
                                               != self.n_components_):
            raise ValueError("Number of input features has changed from %i "
                             "to %i between calls to partial_fit! Try "
                             "setting n_components to a fixed value." % (
                                 self.components_.shape[0], self.n_components_))

        if self.components_ is None:
            # This is the first pass through partial_fit
            self.n_samples_seen_ = 0
            col_var = X.var(axis=0)
            col_mean = X.mean(axis=0)
            X -= col_mean
            U, S, V = linalg.svd(X, full_matrices=False)
            U, V = svd_flip(U, V, u_based_decision=False)
            explained_variance = S ** 2 / n_samples
            explained_variance_ratio = S ** 2 / np.sum(col_var *
                                                       n_samples)
        else:
            col_batch_mean = X.mean(axis=0)
            col_mean, col_var, n_total_samples = _batch_mean_variance_update(
                X, self.mean_, self.var_, self.n_samples_seen_)
            X -= col_batch_mean
            # Build matrix of combined previous basis and new data
            mean_correction = np.sqrt((self.n_samples_seen_ * n_samples) /
                                      n_total_samples) * (self.mean_ -
                                                          col_batch_mean)
            X_combined = np.vstack((self.singular_values_.reshape((-1, 1)) *
                                    self.components_, X,
                                    mean_correction))
            U, S, V = linalg.svd(X_combined, full_matrices=False)
            U, V = svd_flip(U, V, u_based_decision=False)
            explained_variance = S ** 2 / n_total_samples
            explained_variance_ratio = S ** 2 / np.sum(col_var *
                                                       n_total_samples)
        self.n_samples_seen_ += n_samples
        self.components_ = V[:self.n_components_]
        self.singular_values_ = S[:self.n_components_]
        self.mean_ = col_mean
        self.var_ = col_var
        self.explained_variance_ = explained_variance[:self.n_components_]
        self.explained_variance_ratio_ = \
            explained_variance_ratio[:self.n_components_]
        # if self.n_components_ < n_features:          # --- ADJUSTED
        #     self.noise_variance_ = \
        #         explained_variance[self.n_components_:].mean()
        # else:
        #     self.noise_variance_ = 0.

        return self