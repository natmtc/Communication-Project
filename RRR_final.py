
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.metrics import make_scorer, r2_score
from scipy import sparse
from numpy import linalg 

class ReducedRankRegressor(BaseEstimator):
        """
        Reduced Rank Regressor (linear 'bottlenecking' or 'multitask learning')
        Constructor parameters
        ----------------------
        rank : int
            rank constraint.
        reg : float (optional)
            regularization parameter
            (alpha in sklearn.linear_model.Ridge)
        """
        def __init__(self, rank, reg=None):
            self.rank = rank
            self.reg = reg if reg is not None else 0


        def __str__(self):
            return 'Reduced Rank Regressor (rank = {})'.format(self.rank)


        def fit(self, _X, _Y):
            """
            Fit reduced rank regressor to data.
            Parameters
            ----------
            _X : ndarray
                matrix of features with shape (n_samples x n_features)
            _Y : ndarray
                matrix of targets with shape (n_samples x n_target_features)
            Returns
            -------
            Sets attributes needed for prediction and returns None
            """
            if np.ndim(_X) == 1:
                _X = np.reshape(_X, (-1, 1))
            if np.ndim(_Y) == 1:
                _Y = np.reshape(_Y, (-1, 1))

            self.mean_input = _X.mean(axis=0)
            self.mean_output = _Y.mean(axis=0)

            X = _X - self.mean_input
            Y = _Y - self.mean_output

            CXX_inv = np.linalg.pinv((X.T @ X) + self.reg * sparse.eye(X.shape[1]))
            CXY = X.T @ Y
            _U, _S, V = np.linalg.svd(CXY.T @ (CXX_inv @ CXY))
            self.W = V[0:self.rank, :].T
            self.A = (CXX_inv @ (CXY @ self.W)).T

            self.projector_mx = self.A.T @ self.W.T

            return self


        def predict(self, X):
            """
            Predict using a low-rank regressor
            Parameters
            ----------
            X : ndarray
                input matrix of features with shape (n_samples x n_features)
            Returns
            -------
            matrix of predictions with shape (n_samples x n_target_features)
            """
            if np.size(np.shape(X)) == 1:
                X = np.reshape(X, (-1, 1))
            return np.array(((X - self.mean_input) @ self.projector_mx) + self.mean_output)

        @property
        def coef_(self):
            return self.projector_mx.T